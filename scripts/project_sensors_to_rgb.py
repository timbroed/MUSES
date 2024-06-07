import argparse
import os

from tqdm import tqdm

from processing.event_camera_processing import load_event_camera_projection
from processing.lidar_processing import load_lidar_projection
from processing.radar_processing import load_radar_projection
from processing.utils import (load_muses_calibration_data, load_meta_data,
                              rescale_and_shift_image,
                              save_image_to_file)


def rescale_and_save_image(output_file, image, scale_factor=None, shift_factor=None):
    """
    Save the image and rescale it to fit into an uint16 image, if the output file type is 'png'.

    Args:
        output_file (str): The output file path.
        image (np.array): The image to be processed and saved.
        enlarge_points (bool): Whether to enlarge the points in the image.
        scale_factor (int): The scale factor for the image. Default is None.
        shift_factor (int): The shift factor for the image. Default is None.

    Returns:
        None
    """
    # Extract the directory path from the output_file
    output_directory_path = os.path.dirname(output_file)
    os.makedirs(output_directory_path, exist_ok=True)

    # Rescale and shift the image if the output file type is 'png'
    if output_file.endswith('.png') and scale_factor is not None and shift_factor is not None:
        image = rescale_and_shift_image(image, scale_factor, shift_factor)

    # Save the image to a file
    save_image_to_file(output_file, image)

def project_sensors_to_rgb(muses_root, process_event_camera, process_lidar, process_radar,
                           target_shape, motion_compensation, output_folder,
                           enlarge_lidar_points, enlarge_radar_points, enlarge_event_camera_points,
                           output_file_type):
    """
    This method projects sensor data to an RGB frame.
    It processes event camera, lidar, and radar data if requested.
    The projected data is saved to the specified output folder.

    Args:
        muses_root (str): The root directory where the data is located.
        process_event_camera (bool): Process event camera data if True.
        process_lidar (bool): Process lidar data if True.
        process_radar (bool): Process radar data if True.
        target_shape (tuple): The target output shape in pixels.
        motion_compensation (bool): Enable motion compensation.
        output_folder (str): The name of the output folder where the processed data will be saved.
        enlarge_lidar_points (bool): Whether to enlarge the lidar points in the image.
        enlarge_radar_points (bool): Whether to enlarge the radar points in the image.
        enlarge_event_camera_points (bool): Whether to enlarge the event camera points in the image.
        output_file_type (str): The output file type. Can be '.png' or '.npy' or 'npz'.

    Returns:
        None
    """
    assert os.path.exists(muses_root), f"Data root {muses_root} does not exist"

    calib_data = load_muses_calibration_data(muses_root)
    meta_data = load_meta_data(muses_root)

    with tqdm(meta_data.items(), total=len(meta_data), desc="Processing Files") as pbar:
        for entry_name, entry_data in pbar:
            pbar.set_description(f"Processing {entry_name}")

            # Process event camera data if requested
            if process_event_camera:
                path_to_event_camera = entry_data.get('path_to_event_camera')
                event_path = os.path.join(muses_root, path_to_event_camera)
                image = load_event_camera_projection(event_path, calib_data, target_shape, enlarge_event_camera_points)
                output_file = os.path.join(muses_root, output_folder, path_to_event_camera.replace('.h5', output_file_type))
                rescale_and_save_image(output_file, image)

            # Process lidar data if requested
            if process_lidar:
                path_to_lidar = entry_data.get('path_to_lidar')
                lidar_path = os.path.join(muses_root, path_to_lidar)
                image = load_lidar_projection(lidar_path, calib_data, entry_data,
                                              motion_compensation, muses_root,
                                              target_shape, enlarge_lidar_points)
                output_file = os.path.join(muses_root, output_folder, path_to_lidar.replace('.bin', output_file_type))
                rescale_and_save_image(output_file, image, scale_factor=150, shift_factor=100)

            # Process radar data if requested
            if process_radar:
                path_to_radar = entry_data.get('path_to_radar')
                radar_path = os.path.join(muses_root, path_to_radar)
                image = load_radar_projection(radar_path, calib_data, entry_data,
                                              motion_compensation, muses_root,
                                              target_shape, enlarge_radar_points)
                output_file = os.path.join(muses_root, output_folder, path_to_radar.replace('.png', output_file_type))
                rescale_and_save_image(output_file, image, scale_factor=150, shift_factor=100)


if __name__ == "__main__":
    # Argument parser with description
    parser = argparse.ArgumentParser(description='Project sensor data to RGB frame.')
    parser.add_argument('--muses_root', default='data/muses/', help='Root path for muses dataset')
    parser.add_argument('--output_folder', type=str, default='projected_to_rgb', help='Output sub folder')
    parser.add_argument('--event_camera', action='store_true', help='Process event camera data')
    parser.add_argument('--lidar', action='store_true', help='Process lidar data')
    parser.add_argument('--radar', action='store_true', help='Process radar data')
    parser.add_argument('--target_width', type=int, default=1920, help='Target output width in pixels')
    parser.add_argument('--target_height', type=int, default=1080, help='Target output height in pixels')
    parser.add_argument('--motion_compensation', action='store_true',
                        help='Enable motion compensation  (default: False)')
    parser.add_argument('--enlarge_lidar_points', action='store_true',
                        help='Enable points enlargement via dilation (default: False)')
    parser.add_argument('--enlarge_radar_points', action='store_true',
                        help='Enable points enlargement via dilation (default: False)')
    parser.add_argument('--enlarge_event_camera_points', action='store_true',
                        help='Enable points enlargement via dilation (default: False)')
    parser.add_argument('--output_file_type', type=str, default='.npz', choices=['.npz', '.npy', '.png'],
                        help='Output file type (default: .npz). '
                        'Can be ".png" to save a shifted and scaled uint16 image '
                        'or ".npz"/".npy" to save the raw image.')

    args = parser.parse_args()

    if not args.event_camera and not args.lidar and not args.radar:
        raise ValueError("At least one sensor must be selected for processing. "
                         "Add any combination of --event_camera, --lidar, or --radar.")

    sensor_types = ['event_camera', 'radar', 'lidar']
    for sensor in sensor_types:
        if getattr(args, sensor) and not os.path.exists(os.path.join(args.muses_root, sensor)):
            raise ValueError(f"The {sensor} data is not available in the data root folder. "
                             "Please download the data from https://muses.vision.ee.ethz.ch/download "
                             "and unzip the sensors data before proceeding.")

    project_sensors_to_rgb(args.muses_root, args.event_camera, args.lidar, args.radar,
                           (args.target_width, args.target_height), args.motion_compensation,
                           args.output_folder,
                           args.enlarge_lidar_points, args.enlarge_radar_points, args.enlarge_event_camera_points,
                           args.output_file_type)
