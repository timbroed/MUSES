import cv2
import h5py
import hdf5plugin
import numpy as np
import os

from processing.utils import enlarge_points_in_image, rescale_K

def stereo_rectify(event_data, cm1, cm2, R, T, rgb_shape):
    """
    This function rectifies the event data.

    Args:
        event_data (list): The event data.
        cm1 (np.array): The intrinsic parameters of the first camera.
        cm2 (np.array): The intrinsic parameters of the second camera.
        R (np.array): The rotation matrix.
        T (np.array): The translation matrix.
        rgb_shape (tuple): The shape of the RGB image.

    Returns:
        tuple: The rectified event data.
    """
    R1, R2, P1, P2, Q, ROI1, ROI2 = cv2.stereoRectify(cm1, None, cm2, None, rgb_shape, R, T,
                                                      flags=cv2.CALIB_ZERO_DISPARITY)
    # we could use the inverse of Q to get disparity values from lidar points

    # for events, we need the forward mapping
    event_pts = np.stack([event_data[0], event_data[1]], axis=1).astype(float)
    event_pts_remap = cv2.undistortPoints(event_pts, cm2, None, R=R2, P=P2)[:, 0, :].T

    return event_pts_remap, R1, P1


def render(x: np.ndarray, y: np.ndarray, pol: np.ndarray, H: int, W: int) -> np.ndarray:
    """
    This function renders the event data to an image, by accumulating the events in the given time period into the 1st and 2nd channels.

    Args:
        x (np.ndarray): The x coordinates.
        y (np.ndarray): The y coordinates.
        pol (np.ndarray): The polarity.
        H (int): The height of the image.
        W (int): The width of the image.

    Returns:
        np.ndarray: The rendered image.
    """
    assert x.size == y.size == pol.size
    assert H > 0
    assert W > 0
    img = np.full((H, W, 3), fill_value=0, dtype='uint8')
    pol = pol.astype('int')

    idx_pos = (pol == 1)
    idx_neg = (pol == 0)

    data = x + 100000 * y

    unique_combinations_pos, counts_pos = np.unique(data[idx_pos], return_counts=True)
    unique_combinations_neg, counts_neg = np.unique(data[idx_neg], return_counts=True)

    x_pos = unique_combinations_pos % 100000
    y_pos = unique_combinations_pos // 100000

    x_neg = unique_combinations_neg % 100000
    y_neg = unique_combinations_neg // 100000

    img[y_pos, x_pos, 0] = counts_pos
    img[y_neg, x_neg, 1] = counts_neg

    return img


def accumulate_events(event_path, accumulate_over_last_ms=30000):
    """
    This function loads the event camera data for a given time interval.
    The data is accumulated over the last "accumulate_over_last_ms" milliseconds.

    Args:
        event_path (str): The path to the event data.
        accumulate_over_last_ms (int): The number of milliseconds to accumulate over.

    Returns:
        tuple: The accumulated events.
    """
    with h5py.File(event_path, 'r') as f:
        x = f['events/x'][:]
        y = f['events/y'][:]
        t = f['events/t'][:]
        p = f['events/p'][:]
        max_t = np.amax(t)
        min_t = max_t - accumulate_over_last_ms
        mask = t >= min_t
        x = x[mask]
        y = y[mask]
        t = t[mask]
        p = p[mask]

    return x, y, t, p

def undistord_events(P_rgb, R_rgb, event_pts_remap, mtx1):
    """
    This function undistorts the events.

    Args:
        P_rgb (np.array): The intrinsic parameters of the RGB camera.
        R_rgb (np.array): The rotation matrix of the RGB camera.
        event_pts_remap (list): The remapped event points.
        mtx1 (np.array): The intrinsic parameters of the event camera.

    Returns:
        tuple: The undistorted event points.
    """
    event_pts = np.stack(event_pts_remap, axis=1).astype(float)
    event_pts_final = cv2.undistortPoints(event_pts, P_rgb[:, :-1], None, R=np.linalg.inv(R_rgb), P=mtx1)[:, 0, :].T
    x, y = event_pts_final
    x = x.round().astype(int)
    y = y.round().astype(int)
    return x, y

def get_event_calib(calib_data):
    """
    This function extracts the calibration data for the event camera.

    Args:
        calib_data (dict): The dictionary containing the calibration data.

    Returns:
        tuple: The calibration parameters for the event camera.
    """
    # load the calibration parameters
    mtx1 = np.array(calib_data["intrinsics"]["rgb"]["K"])
    mtx2 = np.array(calib_data["intrinsics"]["event"]["K"])
    event2rgb = np.array(calib_data["extrinsics"]["event2rgb"])
    rgb2event = np.linalg.inv(event2rgb)
    rotation_rgb2event = rgb2event[:3, :3]
    translation_rgb2event = rgb2event[:3, 3]
    return mtx1, mtx2, rotation_rgb2event, translation_rgb2event

# Define functions for processing lidar data
def load_points_in_image_event_camera(event_path, calib_data, rgb_width=1920, rgb_height=1080):
    """
    This loads the event camera data and projects it to an RGB frame.

    Args:
        event_path (str): The path to the event camera data.
        rgb_width (int): The width of the RGB image.
        rgb_height (int): The height of the RGB image.
        calib_data (dict): The calibration data.

    Returns:
        tuple: The event camera points in the image. Including x, y, and polarity.
    """
    # Call the new function to accumulate events
    x, y, t, p = accumulate_events(event_path, 30000)

    # load the calibration parameters
    mtx1, mtx2, rotation_rgb2event, translation_rgb2event = get_event_calib(calib_data)

    # rescale the intrinsic parameters of the event camera
    mtx1 = rescale_K(mtx1, rgb_height, rgb_width)

    # the data is already undistorted, from the processing
    # next we can do rectification
    event_pts_remap, R_rgb, P_rgb = stereo_rectify([x, y],
                                                mtx1,
                                                mtx2,
                                                rotation_rgb2event,
                                                translation_rgb2event,
                                                (rgb_width, rgb_height))

    # transform the events back to undistorted view
    x, y = undistord_events(P_rgb, R_rgb, event_pts_remap, mtx1)

    return x, y, p

def load_event_camera_projection(event_path, calib_data, target_shape=(1920, 1080), enlarge_event_camera_points=False):
    """
    This method loads the event camera data and projects it to an RGB frame.
    The events are accumulated over the last 30 milliseconds into the 1st and 2nd channels.
    Optionally, the event camera points can be dilated via the enlarge_event_camera_points flag.

    Args:
        event_path (str): The path to the event camera data.
        calib_data (dict): The calibration data.
        target_shape (tuple): The target output shape in pixels.
        enlarge_event_camera_points (bool): Whether to enlarge the event camera points in the image.

    Returns:
        np.array: The image with the projected event camera points.
    """
    rgb_width, rgb_height = target_shape
    x, y, p = load_points_in_image_event_camera(event_path, calib_data, rgb_width, rgb_height)
    image = render(x, y, p, rgb_height, rgb_width)
    if enlarge_event_camera_points:
        image = enlarge_points_in_image(image, kernel_shape=(2, 2))
    return image