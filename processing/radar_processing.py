import cv2
import math
import numpy as np
import os

from processing.utils import filter_and_project_pcd_to_image, motion_compensate_pcd, create_image_from_point_cloud, enlarge_points_in_image


def extract_timestamps(front_radar_polar_image, i):
    """
    Extract the timestamps from the radar data.

    Args:
        front_radar_polar_image (np.array): The raw azimuth range radar data.
        i (int): The index.

    Returns:
        np.array: The timestamps in microseconds.
    """
    timestamps_s = front_radar_polar_image[:4, i].copy().transpose().reshape(-1).view(np.uint32).astype(np.float64)
    timestamps_ns = front_radar_polar_image[4:8, i].copy().transpose().reshape(-1).view(np.uint32).astype(
        np.float64)
    timestamps_us = (timestamps_s * 1000000.) + (timestamps_ns / 1e9 * 1000000.)  # microseconds
    return timestamps_us


def calculate_coordinates(distance, azimuths, valid_mask, i):
    """
    Calculate the x and y coordinates.

    Args:
        distance (np.array): The distance data.
        azimuths (np.array): The azimuth data.
        valid_mask (np.array): The valid mask.
        i (int): The index.

    Returns:
        np.array: The x and y coordinates.
    """
    x = distance[valid_mask] * math.cos(azimuths[i])
    y = distance[valid_mask] * math.sin(azimuths[i])
    return x, y


def append_to_point_cloud(xyzi, point_count, num_points, x, y, fft_data, valid_mask, timestamps_us):
    """
    Append the radar data to the point cloud.

    Args:
        xyzi (np.array): The point cloud data.
        point_count (int): The current point count.
        num_points (int): The number of points.
        x (np.array): The x coordinates.
        y (np.array): The y coordinates.
        fft_data (np.array): The fft data.
        valid_mask (np.array): The valid mask.
        timestamps_us (np.array): The timestamps in microseconds.

    Returns:
        np.array: The updated point cloud.
    """
    xyzi[point_count:point_count + num_points, 0] = x
    xyzi[point_count:point_count + num_points, 1] = y
    xyzi[point_count:point_count + num_points, 3] = fft_data[valid_mask]
    xyzi[point_count:point_count + num_points, 4] = np.repeat(timestamps_us * 1.e-6, num_points)
    return xyzi


def radar_to_point_cloud(front_radar_polar_image, intensity_threshold=0, image_fov_only=False, step=400,
                         ground_level=1.62):
    """
    Convert radar data to a point cloud.
    This funktion sets the height channel to the ground level (at 1.62m).

    Args:
        front_radar_polar_image (np.array): The raw azimuth range radar data.
        intensity_threshold (int): The intensity threshold for the radar data (default is 0)
        step (int): The step size for the radar data (default is 400)
        image_fov_only (bool): Whether to load points in the image Field of View only (default is False)
        ground_level (float): The ground level (default is 1.62) used for the height channel of the point cloud.

    Returns:
        np.array: The radar data as a point cloud.
    """
    assert front_radar_polar_image.shape[1] == step
    if image_fov_only:
        min_azimuths = 145
        max_azimuths = 255
    else:
        min_azimuths = 0
        max_azimuths = step - 1

    azimuths = -np.radians(np.linspace(-180 + (360 / step), 180, step).astype(np.float32))
    distance = np.linspace(0.0438, 330.0768, front_radar_polar_image.shape[0] - 11)
    range_in_bins = 7536
    max_points = (max_azimuths - min_azimuths + 1) * range_in_bins
    xyzi = np.zeros((max_points, 5), dtype=np.float64)
    point_count = 0

    for i in range(min_azimuths, max_azimuths):
        timestamps_us = extract_timestamps(front_radar_polar_image, i)
        valid_flag = front_radar_polar_image[10, i].copy().view(np.uint8)
        if valid_flag:
            fft_data = front_radar_polar_image[11:, i]
            valid_mask = fft_data >= intensity_threshold
            x, y = calculate_coordinates(distance, azimuths, valid_mask, i)
            num_points = sum(valid_mask)
            xyzi = append_to_point_cloud(xyzi, point_count, num_points, x, y, fft_data, valid_mask, timestamps_us)
            point_count += num_points
        else:
            print("[WARNING] valid_flag 0 encountered. Skipping point cloud conversion for the azimuths.")

    xyzi = xyzi[:point_count, :]
    xyzi[:, 2] -= ground_level  # Set height to ground level

    return xyzi


def load_raw_radar_data(radar_path):
    """
    Load radar data from the specified path.

    Args:
        radar_path (str): The path to the radar data.

    Returns:
        np.array: The raw azimuth range radar data.
    """
    radar_range_azimuth = cv2.imread(radar_path)[:, :, 0]
    return radar_range_azimuth

def load_radar_as_pcd(radar_path, intensity_threshold=0, image_fov_only=False):
    """
    Load radar data as a point cloud.

    Args:
        radar_path (str): The path to the radar data.
        intensity_threshold (int): The intensity threshold for the radar data.
        image_fov_only (bool): Whether to load points in the image Field of View only.

    Returns:
        np.array: The radar data as a point cloud.
    """
    radar_range_azimuth = load_raw_radar_data(radar_path)
    pcd_points = radar_to_point_cloud(radar_range_azimuth,
                                      intensity_threshold=intensity_threshold,
                                      image_fov_only=image_fov_only)
    return pcd_points

def load_points_in_image_radar(radar_path,
                               calib_data,
                               scene_meta_dict=None,
                               motion_compensation=False,
                               muses_root=None,
                               target_shape=(1920, 1080),
                               intensity_threshold=0):
    """
    This method loads radar data and projects it to an RGB frame.

    Args:
        radar_path (str): The path to the radar data.
        calib_data (dict): The calibration data.
        scene_meta_dict (dict): The metadata dictionary for the current data entry.
        motion_compensation (bool): Enable motion compensation.
        muses_root (str): The root directory where the muses dataset is located.
        target_shape (tuple): The target output shape in pixels.
        intensity_threshold (int): The intensity threshold for the radar data.

    Returns:
        None
    """

    assert os.path.exists(radar_path), f"Radar data {radar_path} does not exist"

    pcd_points = load_radar_as_pcd(radar_path, intensity_threshold, image_fov_only=True)

    if motion_compensation:
        assert scene_meta_dict is not None, "Meta_dict must be provided for motion compensation"
        assert muses_root is not None, "MUSES root directory must be provided for motion compensation"
        radar2gnss = np.array(calib_data["extrinsics"]["radar2gnss"])
        pcd_points = motion_compensate_pcd(muses_root, scene_meta_dict, pcd_points, radar2gnss, ts_channel_num=4)

    K_rgb = calib_data["intrinsics"]["rgb"]["K"]
    radar2rgb = calib_data["extrinsics"]["radar2rgb"]
    uv_img_cords_filtered, pcd_points_filtered = filter_and_project_pcd_to_image(pcd_points, radar2rgb, K_rgb,
                                                                                 target_shape, max_distance=150)

    return uv_img_cords_filtered, pcd_points_filtered


def load_radar_projection(radar_path, calib_data, scene_meta_dict=None, motion_compensation=False,
                          muses_root=None, target_shape=(1920, 1080), enlarge_radar_points=False):
    """
    This method loads the radar data and projects it to an RGB frame.
    Optionally, the radar points can be dilated via the enlarge_radar_points flag.

    Args:
        radar_path (str): The path to the radar data.
        calib_data (dict): The calibration data.
        scene_meta_dict (dict): The metadata dictionary for the current data entry.
        motion_compensation (bool): Enable motion compensation.
        muses_root (str): The root directory where the muses dataset is located.
        target_shape (tuple): The target output shape in pixels.
        enlarge_radar_points (bool): Whether to enlarge the radar points in the image.

    Returns:
        np.array: The image with the projected radar points.
    """
    uv_img_cords_filtered, pcd_points_filtered = load_points_in_image_radar(
        radar_path,
        calib_data,
        scene_meta_dict=scene_meta_dict,
        motion_compensation=motion_compensation,
        muses_root=muses_root,
        target_shape=target_shape)
    image = create_image_from_point_cloud(uv_img_cords_filtered, pcd_points_filtered, target_shape,
                                          height_channel=False)
    if enlarge_radar_points:
        image = enlarge_points_in_image(image, kernel_shape=(10, 10))
    return image
