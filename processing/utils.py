import copy
import json
import os

import cv2
import numpy as np
import utm

def load_meta_data(muses_root):
    """
    Load the metadata from the JSON file.

    Args:
        muses_root (str): The root directory where the muses dataset is located.

    Returns:
        dict: The metadata dictionary.
    """
    return read_json_file(os.path.join(muses_root, 'meta.json'))

def read_json_file(file_path):
    """
    Read a JSON file and return its contents.

    Args:
        file_path (str): The path to the JSON file.

    Returns:
        dict: The contents of the JSON file.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File '{file_path}' does not exist.")

    with open(file_path, 'r') as json_file:
        return json.load(json_file)

def read_gnss_file(load_path):
    """
    Read a GNSS file and return a dictionary with the file values.

    Args:
        load_path (str): The path to the GNSS file.

    Returns:
        dict: The GNSS data.
    """
    keys = ['iTOW', 'timestamp', 'tAcc', 'lon', 'lat', 'height', 'hMSL', 'fixType', 'numSV', 'hAcc', 'vAcc', 'roll', \
            'pitch', 'heading', 'accRoll', 'accPitch', 'accHeading', \
            'angular_rate_roll', 'angular_rate_pitch', 'angular_rate_heading', \
            'velN', 'velE', 'velD', 'gSpeed', 'sAcc', 'pDOP', \
            'magDec', 'magAcc']
    with open(load_path, "r", encoding="utf-8") as g:
        values_str = g.readline().split()
        values = [int(x) for x in values_str]
    assert len(keys) == len(values)
    out_dict = {}
    for i, key in enumerate(keys):
        out_dict[key] = values[i]
    return out_dict


def load_muses_calibration_data(input_dir, file_name="calib.json", to_numpy=True):
    """
    Loads calibration data from a JSON file.

    Args:
        input_dir (str): The directory containing the calibration file.
        file_name (str): The name of the calibration file. Default is "calib.json".
        to_numpy (bool): Whether to convert the calibration data to NumPy arrays. Default is True.

    Returns:
        dict: The calibration data.
    """
    params = json.load(open(os.path.join(input_dir, file_name), "rb"))
    if to_numpy:
        for camera in params['intrinsics'].keys():
            for matrix in params['intrinsics'][camera].keys():
                params['intrinsics'][camera][matrix] = np.array(params['intrinsics'][camera][matrix])
        for tranform in params['extrinsics'].keys():
            params['extrinsics'][tranform] = np.array(params['extrinsics'][tranform])
    return params


def create_image_from_point_cloud(uv_img_cords_filtered, filtered_pcd_points, target_shape=(1920, 1080),
                                        height_channel=True, dtype=np.float32):
    """
    Create a float image from a point cloud.

    Args:
        uv_img_cords_filtered (np.array): The filtered UV image coordinates.
        filtered_pcd_points (np.array): The filtered point cloud points.
        target_shape (tuple): The target image shape in pixels.
        height_channel (bool): Whether to include the height channel in the image. Default is True.
        dtype (np.dtype): The data type for the image. Default is np.float32.

    Returns:
        np.array: The float image created from the point cloud.
    """
    w, h = target_shape
    assert uv_img_cords_filtered.shape[1] == filtered_pcd_points.shape[0], "Dimensions mismatch."

    # Initialize an image with zeros
    image = np.zeros((h, w, 3), dtype=dtype)

    # Extract x, y, z, intensity, ring, and timestamp from filtered_pcd_points
    x = filtered_pcd_points[:, 0]  # front
    y = filtered_pcd_points[:, 1]  # left-right
    z = filtered_pcd_points[:, 2]  # height
    intensity = filtered_pcd_points[:, 3]

    # Calculate range as the Euclidean distance
    range_channel = np.sqrt(x ** 2 + y ** 2 + z ** 2)

    # Height is the z coordinate
    height_channel_entry = z if height_channel else np.zeros(z.shape)

    # Map and store the channels in the image
    image[uv_img_cords_filtered[1, :].astype(int), uv_img_cords_filtered[0, :].astype(int), 0] = range_channel
    image[uv_img_cords_filtered[1, :].astype(int), uv_img_cords_filtered[0, :].astype(int), 1] = intensity
    image[uv_img_cords_filtered[1, :].astype(int), uv_img_cords_filtered[0, :].astype(int), 2] = height_channel_entry

    return image

def rescale_and_shift_image(image, scale_factor=150, shift_factor=100):
    """
    Rescale and shift the data in the image and convert it to uint16.

    Args:
        image (np.array): The input float image.
        scale_factor (int): The scale factor for the image. Default is 150.
        shift_factor (int): The shift factor for the image. Default is 100.

    Returns:
        np.array: The rescaled and shifted image in uint16 format.
    """
    # Rescale and shift the image
    image = (image + shift_factor) * scale_factor

    assert (image >= 0).all() and (image <= 65535).all(), "Values should be within the valid range for uint16."

    # Ensure the values are within the valid range for uint16
    image = np.clip(image, 0, 65535)

    # Convert the image to uint16
    image = image.astype(np.uint16)

    return image


def filter_points_by_distance(point_cloud, min_distance=1.0, max_distance=None):
    """

    Filter a point cloud based on minimum and maximum distances from the origin (0, 0, 0).

    Args:
        point_cloud (np.array): The 3D array representing a point cloud, where each row contains the coordinates of a point in 3D spac.
        min_distance (float): The minimum distance from the origin. Points closer than this will be filtered out. Default is 1.0.
        max_distance (float): The maximum distance from the origin. Points farther than this will be filtered out. If not provided, only the minimum distance is considered. Default is None.

    Returns:
        np.array: The filtered point cloud.
    """
    # Calculate the Euclidean distance of each point from the origin (0, 0, 0)
    distances = np.sqrt(np.sum(point_cloud[:, :3] ** 2, axis=1))

    # Find indices of points that are at least min_distance away
    indices = np.where(distances >= min_distance)

    # Filter the point cloud to keep only the points meeting the distance criterion
    filtered_point_cloud = point_cloud[indices]

    if max_distance:
        indices = np.where(distances <= max_distance)
        filtered_point_cloud = filtered_point_cloud[indices]

    return filtered_point_cloud


def filter_and_project_pcd_to_image(pcd, sensor2rgb, K_rgb, target_shape=(1920, 1080), min_distance=1.0, max_distance=None):
    """
    Projects a point cloud to an image and filters the points based on distance and image boundaries.

    Args:
        pcd (np.array): The point cloud.
        sensor2rgb (np.array): The transformation matrix from sensor to RGB coordinates.
        K_rgb (np.array): The camera intrinsic matrix.
        target_shape (tuple): The target image shape (width, height) in pixels .
        min_distance (float): The minimum distance from the origin. Points closer than this will be filtered out. Default is 1.0.
        max_distance (float): The maximum distance from the origin. Points farther than this will be filtered out. If not provided, only the minimum distance is considered. Default is None.

    Returns:
        tuple: The filtered UV image coordinates and the filtered point cloud.
    """
    pcd = filter_points_by_distance(pcd, min_distance=min_distance, max_distance=max_distance)

    point_cloud_xyz = pcd[:, :3]
    w, h = target_shape

    K_rgb = rescale_K(K_rgb, h, w)

    uv_img_cords = project_pcd_to_image(K_rgb, point_cloud_xyz, sensor2rgb)

    pcd, uv_img_cords_filtered = filter_by_image_boundaries(pcd, uv_img_cords, h, w)

    return uv_img_cords_filtered, pcd


def project_pcd_to_image(K_rgb, point_cloud_xyz, sensor2rgb):
    """
    Project a point cloud to an image using the camera intrinsic matrix and the transformation matrix from sensor to RGB coordinates.

    Args:
        K_rgb (np.array): The camera intrinsic matrix.
        point_cloud_xyz (np.array): The point cloud.
        sensor2rgb (np.array): The transformation matrix from sensor to RGB coordinates.

    Returns:
        np.array: The UV image coordinates.
    """
    pcnp_front_ones = np.concatenate((point_cloud_xyz, np.ones((point_cloud_xyz.shape[0], 1))), axis=1)
    point_cloud_xyz_front = sensor2rgb @ pcnp_front_ones.T
    uv_img_cords = K_rgb @ point_cloud_xyz_front[0:3, :] / point_cloud_xyz_front[2, :]
    return uv_img_cords


def filter_by_image_boundaries(pcd_lidar, uv_img_cords, h, w):
    """
    Filter the point cloud and UV image coordinates based on the image boundaries.

    Args:
        pcd_lidar (np.array): The point cloud.
        uv_img_cords (np.array): The UV image coordinates.
        h (int): The image height.
        w (int): The image width.

    Returns:
        tuple: The filtered point cloud and UV image coordinates.
    """
    valid_indices = (
            (uv_img_cords[0, :] > 0) &
            (uv_img_cords[0, :] < (w - 1)) &
            (uv_img_cords[1, :] > 0) &
            (uv_img_cords[1, :] < (h - 1))
    )
    pcd_lidar = pcd_lidar[valid_indices]
    uv_img_cords_filtered = uv_img_cords[:, valid_indices]
    return pcd_lidar, uv_img_cords_filtered


def rescale_K(K_rgb, h, w, previous_h=1920, previous_w=1080):
    """
    Rescales the given camera intrinsic matrix based on the new image dimensions.

    Args:
        K_rgb (np.array): The camera intrinsic matrix.
        h (int): The new image height.
        w (int): The new image width.
        previous_h (int): The previous image height. Default is 1920.
        previous_w (int): The previous image width. Default is 1080.

    Returns:
        np.array: The rescaled camera intrinsic matrix.
    """
    scale_x = w / previous_h
    scale_y = h / previous_w
    K_rgb = np.array(K_rgb, dtype=np.float32)
    K_rgb[0, 0] *= scale_x  # fx
    K_rgb[1, 1] *= scale_y  # fy
    K_rgb[0, 2] *= scale_x  # cx
    K_rgb[1, 2] *= scale_y  # cy
    return K_rgb


def motion_compensate_pcd(muses_root, scene_meta_data, pcd_points, sensor2gnss, ts_channel_num):
    """
    Main function to perform motion compensation on Point Cloud (PCD) data.
    - It loads GNSS data, calculates time difference, and applies transformations on the PCD points.

    Args:
        muses_root (str): The root directory where the muses dataset is located.
        scene_meta_data (dict): The metadata dictionary for the current data entry.
        pcd_points (np.array): The point cloud data.
        sensor2gnss (np.array): The transformation matrix from sensor to GNSS coordinates.
        ts_channel_num (int): The channel number for the timestamp in the PCD points.

    Returns:
        np.array: The corrected PCD points in the sensor frame.
    """
    # Load the GNSS data
    ublox_data = load_gnss_data(muses_root, scene_meta_data)

    # Calculate the time difference for each point
    delta_ts_in_s = calculate_time_diff(pcd_points, scene_meta_data, ts_channel_num)

    # Add the transformation from sensor frame to GNSS frame
    pcd_points_in_gnss_frame = apply_transformation(pcd_points, sensor2gnss)

    # Apply correction for the ego-motion in GNSS frame
    corrected_pcd_points_in_gnss_frame = apply_correction(pcd_points_in_gnss_frame, ublox_data, delta_ts_in_s)

    # Transform the corrected points back to sensor's frame
    gnss2sensor = np.linalg.inv(sensor2gnss)
    corrected_pcd_points_in_sensor_frame = apply_transformation(corrected_pcd_points_in_gnss_frame, gnss2sensor)

    return corrected_pcd_points_in_sensor_frame


def load_gnss_data(muses_root, scene_meta_data):
    """
    Loads the GNSS data from file.
    - Constructs the path using `muses_root` and `meta_dict`.
    - Invokes `read_gnss_file` function using the constructed path.

    Args:
        muses_root (str): The root directory where the muses dataset is located.
        scene_meta_data (dict): The metadata dictionary for the current data entry.

    Returns:
        dict: The GNSS data.
    """
    gnss_path = os.path.join(muses_root, scene_meta_data['path_to_gnss'])
    load_path = os.path.join(muses_root, gnss_path)
    return read_gnss_file(load_path)


def calculate_time_diff(pcd_points, scene_meta_data, ts_channel_num):
    """
    Calculates the time difference for each PCD point.
    - Uses the exposure start/end timestamps in `meta_dict` to compute the target timestamp.
    - Subtracts target timestamp from PCD points timestamps to calculate time difference.

    Args:
        pcd_points (np.array): The point cloud data.
        scene_meta_data (dict): The metadata dictionary for the current data entry.
        ts_channel_num (int): The channel number for the timestamp in the PCD points.

    Returns:
        np.array: The time difference for each PCD point.
    """
    rgb_start_ts = int(scene_meta_data['frame_camera_exposure_start_timestamp_us'])
    rgb_end_ts = int(scene_meta_data['frame_camera_exposure_end_timestamp_us'])
    target_ts = (rgb_start_ts + rgb_end_ts) / 2
    target_ts_s = target_ts / 1e6
    return pcd_points[:, ts_channel_num] - target_ts_s


def apply_correction(pcd_points, ublox_data, delta_ts_in_s):
    """
    Applies linear and rotational transformations on PCD points.
    - Applies linear transformation induced by vehicle speed.
    - Applies rotational transformation induced by change in roll, pitch and heading.

    Args:
        pcd_points (np.array): The point cloud data.
        ublox_data (dict): The GNSS data.
        delta_ts_in_s (np.array): The time difference for each PCD point.

    Returns:
        np.array: The corrected PCD points.
    """
    pcd_points = apply_linear_correction(pcd_points, ublox_data, delta_ts_in_s)
    pcd_points = apply_rotational_correction(pcd_points, ublox_data, delta_ts_in_s)
    return pcd_points


def apply_linear_correction(pcd_points, ublox_data, delta_ts_in_s):
    """
    Applies linear transformation on PCD points, assuming planar motion of the car in x direction.
    - Uses vehicle speed data from GNSS for linear correction.

    Args:
        pcd_points (np.array): The point cloud data.
        ublox_data (dict): The GNSS data.
        delta_ts_in_s (np.array): The time difference for each PCD point.

    Returns:
        np.array: The corrected PCD points.
    """
    x_correction_mm = ublox_data['gSpeed'] * delta_ts_in_s
    pcd_points[:, 0] += x_correction_mm / 1e3
    return pcd_points



def get_rotation_matrix(roll, pitch, yaw):
    """
    Constructs a rotation matrix from roll, pitch, and yaw (in radians).
    """
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(roll), -np.sin(roll)],
                   [0, np.sin(roll), np.cos(roll)]])

    Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                   [0, 1, 0],
                   [-np.sin(pitch), 0, np.cos(pitch)]])

    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                   [np.sin(yaw), np.cos(yaw), 0],
                   [0, 0, 1]])

    return Rz @ Ry @ Rx

def apply_rotational_correction(pcd_points, ublox_data, delta_ts_in_s):
    """
    Applies rotational transformation on PCD points.
    - Systematically corrects roll, pitch and heading based on GNSS data.
    - Rotational correction is applied on each PCD point.

    Args:
        pcd_points (np.array): The point cloud data.
        ublox_data (dict): The GNSS data.
        delta_ts_in_s (np.array): The time difference for each PCD point.

    Returns:
        np.array: The corrected PCD points.
    """
    roll_correction = np.deg2rad(ublox_data['angular_rate_roll'] * delta_ts_in_s / 1e5)
    pitch_correction = np.deg2rad(ublox_data['angular_rate_pitch'] * delta_ts_in_s / 1e5)
    yaw_correction = np.deg2rad(ublox_data['angular_rate_heading'] * delta_ts_in_s / 1e5)

    # Calculate sin and cos for each correction
    sr, cr = np.sin(roll_correction), np.cos(roll_correction)
    sp, cp = np.sin(pitch_correction), np.cos(pitch_correction)
    sy, cy = np.sin(yaw_correction), np.cos(yaw_correction)

    # Rotation matrix components
    R11 = cy * cp
    R12 = cy * sp * sr - sy * cr
    R13 = cy * sp * cr + sy * sr
    R21 = sy * cp
    R22 = sy * sp * sr + cy * cr
    R23 = sy * sp * cr - cy * sr
    R31 = -sp
    R32 = cp * sr
    R33 = cp * cr

    # Apply the rotation matrices
    x, y, z = pcd_points[:, 0], pcd_points[:, 1], pcd_points[:, 2]
    pcd_points[:, 0] = R11 * x + R12 * y + R13 * z
    pcd_points[:, 1] = R21 * x + R22 * y + R23 * z
    pcd_points[:, 2] = R31 * x + R32 * y + R33 * z

    return pcd_points


def apply_transformation(pcd_points, transformation_matrix):
    """
    Applies a transformation on the PCD points.

    Args:
        pcd_points (np.array): The point cloud data.
        transformation_matrix (np.array): The 4x4 transformation matrix.

    Returns:
        np.array: The transformed PCD points.
    """
    # Add a 1 to each 3D point to have homogeneous coordinates
    homogeneous_pcd_points = np.hstack((pcd_points[:, :3], np.ones((len(pcd_points), 1))))

    # Apply the transformation matrix
    transformation_matrix_np = np.array(transformation_matrix)
    homogeneous_transformed_pcd_points = transformation_matrix_np @ homogeneous_pcd_points.T
    transformed_pcd_points = homogeneous_transformed_pcd_points.T[:, :3]

    pcd_points[:, :3] = transformed_pcd_points

    return pcd_points


def save_image_to_file(output_file, image):
    """
    Save an image to a file.

    Args:
        output_file (str): The name of the output file of type .png, .npy, or .npz.
        image (np.array): The image to be saved.

    Returns:
        None
    """
    if output_file.endswith('.png'):
        cv2.imwrite(output_file, image)
    elif output_file.endswith('.npy'):
        np.save(output_file, image)
    elif output_file.endswith('.npz'):
        np.savez_compressed(output_file, image)
    else:
        raise ValueError("Invalid file type.")


def normalize_and_mask_image(image):
    """
    Normalize and mask the input image.

    Args:
        image (np.array): The input image to be normalized and masked.

    Returns:
        np.array: The normalized and masked image.
    """
    normalized_img = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    normalized_img = cv2.cvtColor(normalized_img, cv2.COLOR_RGB2BGR)
    mask = (normalized_img == 12)
    normalized_img[mask] = 0
    return normalized_img


def enlarge_points_in_image(image, kernel_shape=(3, 3)):
    """
    Enlarge points in an image by performing a dilation operation.

    Args:
        image (np.array): The input image.
        kernel_shape (tuple): The shape of the kernel for the dilation operation. Default is (3, 3).

    Returns:
        np.array: The image with enlarged points.
    """
    kernel = np.ones(kernel_shape, np.uint8)
    return cv2.dilate(image, kernel, iterations=1)


def points_to_world_coord_ned(pcd_points,
                              ublox_data,
                              sensor2gnss,
                              origin_ublox_data=None):
    """
    Convert points from a local coordinate system to the NED (North-East-Down) coordinate system.

    Args:
        pcd_points (np.array): The point cloud in the local coordinate system.
        ublox_data (dict): The GNSS data.
        sensor2gnss (np.array): The transformation matrix from sensor to GNSS coordinates.
        origin_ublox_data (dict): The reference GNSS data used as the origin for NED coordinates. If not provided, the
                              ublox_data will be used as the origin. Default is None.

    Returns:
        np.array: The points in the NED coordinate system.
    """
    if not origin_ublox_data:
        origin_ublox_data = ublox_data
    gnss2ned = ublox_to_gnss2ned(ublox_data, origin_ublox_data)
    # Convert to GNSS coordinate system (as frd[forward, right, down])
    pcd_points_in_gnss = apply_transformation(pcd_points, sensor2gnss)
    # Convert to NED coordinate system
    pcd_points_in_ned_frame = apply_transformation(pcd_points_in_gnss, gnss2ned)
    return pcd_points_in_ned_frame, origin_ublox_data


def ublox_to_gnss2ned(ublox_data, reference_point_ublox_data=None):
    """
    Convert ublox data to GNSS to NED (North-East-Down) transformation matrix.

    Args:
        ublox_data (dict): The GNSS data: A dictionary containing 'roll', 'pitch', and 'heading' keys with corresponding values in deg * 1e-5 format.
        reference_point_ublox_data (dict): The reference GNSS data used as the origin for NED coordinates. If not provided, the
                                ublox_data will be used as the origin. Default is None.

    Returns:
        np.array: The transformation matrix from GNSS to NED coordinates.
    """
    ublox_roll = ublox_data['roll'] / 1e5  # Convert from deg * 1e-5 to deg
    ublox_pitch = ublox_data['pitch'] / 1e5  # Convert from deg * 1e-5 to deg
    ublox_heading = ublox_data['heading'] / 1e5  # Convert from deg * 1e-5 to deg

    # Get rotation matrix
    rotation_gnss2ned = get_rotation_gnss2ned(ublox_roll, ublox_pitch, ublox_heading) # GPS_to_NED
    translation_gnss2ned = get_translation_gnss2ned(ublox_data, reference_point_ublox_data)

    # Embedding the rotation matrix in a homogeneous transformation matrix
    gnss2ned = np.eye(4)
    gnss2ned[:3, :3] = rotation_gnss2ned
    gnss2ned[:3, 3] = translation_gnss2ned
    return gnss2ned

def get_translation_gnss2ned(ublox_data, reference_point_ublox_data=None):
    """
    Calculates the translation vector from GNSS coordinates to NED (North-East-Down) coordinates.

    Args:
        ublox_data (dict): The GNSS data including (lon, lat, height) in u-blox format
        reference_point_ublox_data (dict): The reference GNSS data used as the origin for NED coordinates. If not provided, the
                                ublox_data will be used as the origin. Default is None.

    Returns:
        np.array: The translation vector from GNSS to NED coordinates.
    """
    if reference_point_ublox_data is None:
        reference_point = np.zeros((3, 1))
    else:
        reference_point_lon = reference_point_ublox_data['lon'] / 1e7  # Convert from deg * 1e-7 to deg
        reference_point_lat = reference_point_ublox_data['lat'] / 1e7  # Convert from deg * 1e-7 to deg
        reference_point_height = reference_point_ublox_data['height'] / 1e3  # Convert from mm to meters
        reference_point = latlon_to_ned(reference_point_height, reference_point_lat, reference_point_lon)

    ublox_lon = ublox_data['lon'] / 1e7  # Convert from deg * 1e-7 to deg
    ublox_lat = ublox_data['lat'] / 1e7  # Convert from deg * 1e-7 to deg
    ublox_height = ublox_data['height'] / 1e3  # Convert from mm to meters

    gps_point_meters = latlon_to_ned(ublox_height, ublox_lat, ublox_lon)

    # Deduct reference point
    gps_point_meters_relative = gps_point_meters - reference_point

    return gps_point_meters_relative.squeeze()


def latlon_to_ned(ublox_height, ublox_lat, ublox_lon):
    """
    Converts the latitude longitude and height from the GNSS to NED (North-East-Down) coordinates

    Args:
        ublox_height (float): The height in meters.
        ublox_lat (float): The latitude coordinate.
        ublox_lon (float): The longitude coordinate.

    Returns:
        np.array: The NED coordinates.
    """
    easting, northing, utm_zone = latlon_to_utmcm(ublox_lat, ublox_lon)
    down = -ublox_height
    gps_point_meters = np.array([northing, easting, down]).reshape((3, 1))
    return gps_point_meters


def get_rotation_gnss2ned(roll, pitch, heading, degrees=True):
    """
    This method calculates the rotation matrix from GNSS (Global Navigation Satellite System)
    to NED (North-East-Down) coordinates. It takes in the roll, pitch, and heading angles in degrees
    and returns the rotation matrix using ZYX euler angles.
    Similar to o3d.geometry.get_rotation_matrix_from_zyx()

    Args:
        roll (float): The roll angle in degrees.
        pitch (float): The pitch angle in degrees.
        heading (float): The heading angle in degrees.
        degrees (bool): Whether the angles are in degrees (True) or radians (False). Default is True.

    Returns:
        np.array: The rotation matrix from GNSS to NED coordinates.
    """
    if degrees:
        roll = np.deg2rad(roll)
        pitch = np.deg2rad(pitch)
        heading = np.deg2rad(heading)

    C_x = np.array([[1, 0, 0],
                    [0, np.cos(roll), np.sin(roll)],
                    [0, -np.sin(roll), np.cos(roll)]])

    C_y = np.array([[np.cos(pitch), 0, -np.sin(pitch)],
                    [0, 1, 0],
                    [np.sin(pitch), 0, np.cos(pitch)]])

    C_z = np.array([[np.cos(heading), np.sin(heading), 0],
                    [-np.sin(heading), np.cos(heading), 0],
                    [0, 0, 1]])

    rotation_gnss2ned = C_z.T.dot(C_y.T).dot(C_x.T)

    return rotation_gnss2ned

def latlon_to_utmcm(latitude, longitude):
    """
    Converts latitude and longitude to UTM zone, easting, northing, and hemisphere.

    Args:
        latitude (float): The latitude coordinate.
        longitude (float): The longitude coordinate.

    Returns:
        tuple: The easting, northing, UTM zone.
    """
    # Convert latitude and longitude to UTM zone, easting, northing and hemisphere
    easting, northing, zone_number, zone_letter = utm.from_latlon(latitude, longitude)
    return easting, northing, f"{zone_number}{zone_letter}"

