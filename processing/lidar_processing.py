import numpy as np
import os

from processing.utils import filter_and_project_pcd_to_image, motion_compensate_pcd, create_image_from_point_cloud, enlarge_points_in_image

def load_lidar_data(lidar_path):
    """
    Load the lidar data from the .bin file.

    Args:
        lidar_path (str): The path to the lidar data.

    Returns:
        np.array: The loaded lidar data.
    """
    loaded_pcd = np.fromfile(lidar_path, dtype=np.float64)
    return loaded_pcd.reshape((-1, 6))

# Define functions for processing lidar data
def load_points_in_image_lidar(lidar_path,
                               calib_data,
                               scene_meta_data=None,
                               motion_compensation=False,
                               muses_root=None,
                               target_shape=(1920, 1080),
                               ):
    """
    This method loads lidar data and projects it to an RGB frame.

    Args:
        lidar_path (str): The path to the lidar data.
        calib_data (dict): The calibration data.
        motion_compensation (bool): Enable motion compensation.
        scene_meta_data (dict): The metadata dictionary for the current data entry.
        target_shape (tuple): The target output shape in pixels.

    Returns:
        None
    """
    assert os.path.exists(lidar_path), f"Lidar data {lidar_path} does not exist"

    # Load the lidar data from the .bin file
    pcd_points = load_lidar_data(lidar_path)

    # Check if motion compensation is enabled
    if motion_compensation:
        assert scene_meta_data is not None, "Scene meta data is required for motion compensation"
        assert muses_root is not None, "MUSES root directory is required for motion compensation"
        lidar2gnss = calib_data["extrinsics"]["lidar2gnss"]
        pcd_points = motion_compensate_pcd(muses_root, scene_meta_data, pcd_points, lidar2gnss, ts_channel_num=5)

    K_rgb = calib_data["intrinsics"]["rgb"]["K"]
    lidar2rgb = calib_data["extrinsics"]["lidar2rgb"]
    uv_img_cords_filtered, pcd_points_filtered = filter_and_project_pcd_to_image(pcd_points, lidar2rgb, K_rgb,
                                                                                 target_shape)
    return uv_img_cords_filtered, pcd_points_filtered


def load_lidar_projection(lidar_path, calib_data, scene_meta_dict=None, motion_compensation=False,
                          muses_root=None, target_shape=(1920, 1080), enlarge_lidar_points=False):
    """
    This method loads the lidar data and projects it to an RGB frame.
    Optionally, the lidar points can be dilated via the enlarge_lidar_points flag.

    Args:
        lidar_path (str): The path to the lidar data.
        calib_data (dict): The calibration data.
        scene_meta_dict (dict): The metadata dictionary for the current data entry.
        motion_compensation (bool): Enable motion compensation.
        muses_root (str): The root directory where the muses dataset is located.
        target_shape (tuple): The target output shape in pixels.
        enlarge_lidar_points (bool): Whether to enlarge the lidar points in the image.

    Returns:
        np.array: The image with the projected lidar points.
    """
    uv_img_cords_filtered, pcd_points_filtered = load_points_in_image_lidar(
        lidar_path,
        calib_data,
        scene_meta_data=scene_meta_dict,
        motion_compensation=motion_compensation,
        muses_root=muses_root,
        target_shape=target_shape)
    image = create_image_from_point_cloud(uv_img_cords_filtered, pcd_points_filtered, target_shape)
    if enlarge_lidar_points:
        image = enlarge_points_in_image(image, kernel_shape=(2, 2))
    return image