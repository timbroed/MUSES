import argparse
import os

from tqdm import tqdm

from processing.lidar_processing import load_lidar_data
from processing.radar_processing import load_radar_as_pcd
from processing.utils import motion_compensate_pcd, load_muses_calibration_data, load_meta_data


def process_ego_motion(dataset_dir, output_dir, process_lidar, process_radar,
                       output_file_type='.bin', output_folder='motion_compensated',
                       radar_threshold=0, radar_fov='all'):
    """
    This method performs ego motion compensation on the lidar and radar point clouds.
    The corrected point clouds are saved to the output directory in a folder named output_folder as a binary file.

    Args:
        dataset_dir (str): The path to the directory containing the dataset.
        output_dir (str): The path to the output directory.
        process_lidar (bool): Process lidar data if True.
        process_radar (bool): Process radar data if True.
        output_file_type (str): The output file type for the motion compensated point cloud (default: .bin).
        output_folder (str): The name of the output folder where the processed data will be saved.

    Returns:
        None
    """

    # Load metadata from JSON
    calib_data = load_muses_calibration_data(dataset_dir)
    meta_data = load_meta_data(dataset_dir)

    if process_lidar:
        print('[INFO] Starting ego motion correcting of the lidar point clouds.')

        for _, meta_dict in tqdm(meta_data.items()):
            lidar_path = os.path.join(dataset_dir, meta_dict['path_to_lidar'])
            lidar_points = load_lidar_data(lidar_path)
            lidar2gnss = calib_data["extrinsics"]["lidar2gnss"]
            lidar_pcd_compensated = motion_compensate_pcd(dataset_dir, meta_dict, lidar_points, lidar2gnss, ts_channel_num=5)

            # Save corrected point cloud
            output_lidar_path = os.path.join(output_folder, meta_dict['path_to_lidar'].replace('.bin', output_file_type))

            save_path = os.path.join(output_dir, output_lidar_path)
            if not os.path.exists(os.path.dirname(save_path)):
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
            lidar_pcd_compensated.tofile(save_path)

        print(f'[INFO] Finished ego motion correcting of the point clouds.')
        print(f'[INFO] Corrected point clouds saved to {output_dir}.')

    if process_radar:
        print('[INFO] Starting ego motion correcting of the radar point clouds.')

        front_fov_only = False if radar_fov == 'all' else True

        for _, meta_dict in tqdm(meta_data.items()):
            radar_path = os.path.join(dataset_dir, meta_dict['path_to_radar'])
            radar_points = load_radar_as_pcd(radar_path, intensity_threshold=radar_threshold,
                                             image_fov_only=front_fov_only)

            radar2gnss = calib_data["extrinsics"]["radar2gnss"]
            radar_points_compensated = motion_compensate_pcd(dataset_dir, meta_dict, radar_points,
                                                              radar2gnss, ts_channel_num=4)

            # Save corrected point cloud
            output_radar_path = os.path.join(output_folder, meta_dict['path_to_radar'].replace('.png', output_file_type))
            save_path = os.path.join(output_dir, output_radar_path)
            if not os.path.exists(os.path.dirname(save_path)):
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
            radar_points_compensated.tofile(save_path)

        print(f'[INFO] Finished ego motion correcting of the point clouds.')
        print(f'[INFO] Corrected point clouds saved to {output_dir}.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Perform ego motion correction on lidar point clouds.")
    parser.add_argument('--muses_root', default='data/muses/', help='Root path for muses dataset')
    parser.add_argument('--output-dir', type=str, help="Path to the output directory. If not provided, it will use the dataset directory.")
    parser.add_argument('--lidar', action='store_true', help='Process lidar data')
    parser.add_argument('--radar', action='store_true', help='Process radar data')
    parser.add_argument('--radar_threshold', type=int, default=0,
                        help='Intensity threshold for the radar. All points below this threshold will be ignored.')
    parser.add_argument('--radar_fov', type=str, default='all', choices=['all', 'image'],
                        help='Radar Field of View (default: all). '
                             'To keep only the points in the RGB images FoV, select image.')
    parser.add_argument('--output_file_type', type=str, default='.bin', choices=['.bin', '.npz', '.npy'],
                        help='Output file type for the motion compensated point cloud (default: .bin).')
    parser.add_argument('--output_folder', type=str, default='motion_compensated',
                        help='The name of the output folder where the motion compensated data will be saved (default: motion_compensated).')
    args = parser.parse_args()

    sensor_types = ['radar', 'lidar']
    for sensor in sensor_types:
        if getattr(args, sensor) and not os.path.exists(os.path.join(args.muses_root, sensor)):
            raise ValueError(f"The {sensor} data is not available in the data root folder. "
                             "Please download the data from https://muses.vision.ee.ethz.ch/download "
                             "and unzip the sensors data before proceeding.")

    if not args.output_dir:
        args.output_dir = args.muses_root

    process_ego_motion(args.muses_root, args.output_dir, args.lidar, args.radar, args.output_file_type, args.output_folder,
                       args.radar_threshold, args.radar_fov)
