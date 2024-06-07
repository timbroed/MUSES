import argparse
import open3d as o3d
import os
from tqdm import tqdm

from processing.lidar_processing import load_lidar_data
from processing.radar_processing import load_radar_as_pcd
from processing.utils import load_muses_calibration_data, motion_compensate_pcd, \
    read_json_file, load_gnss_data, \
    points_to_world_coord_ned

def visualize_3d(args):
    """
    This method visualizes the 3D data from the MUSES dataset.
    It loads the lidar and radar data, performs motion compensation and visualizes them in the NED frame.

    Args:
        args (argparse.Namespace): The arguments to configure the visualization.

    Returns:
        None
    """
    data_root = args.muses_root
    lidar_vis = args.lidar
    radar_vis = args.radar
    radar_threshold = args.radar_threshold
    front_fov_only = not args.full_radar_fov

    # Load calibration data
    calib_data = load_muses_calibration_data(data_root)

    # Create full path to JSON file
    json_file_path = os.path.join(data_root, 'meta.json')

    data = read_json_file(json_file_path)

    if args.split:
        print(f"Filtering data by split: {args.split}")
        data = {key: value for key, value in data.items() if value['split'] == args.split}
    if args.time_of_day:
        print(f"Filtering data by time of day: {args.time_of_day}")
        data = {key: value for key, value in data.items() if value['time_of_day'] == args.time_of_day}
    if args.weather:
        print(f"Filtering data by weather: {args.weather}")
        data = {key: value for key, value in data.items() if value['weather'] == args.weather}

    origin_ublox_data = None

    # Use tqdm for progress bar
    with tqdm(data.items(), desc="Processing Files") as pbar:
        # Iterate over each entry in the JSON
        for idx, (entry_name, entry_data) in enumerate(pbar):
            pbar.set_description(f"Processing {entry_name}")

            point_clouds_ned = []

            if lidar_vis:
                # Extract relevant data fields
                lidar_path = os.path.join(data_root, entry_data.get('path_to_lidar'))

                # Load lidar data from .bin file
                pcd_points = load_lidar_data(lidar_path)

                lidar2gnss = calib_data["extrinsics"]["lidar2gnss"]
                if args.motion_compensation:
                    pcd_points = motion_compensate_pcd(data_root, entry_data,
                                                                   pcd_points, lidar2gnss,
                                                                   ts_channel_num=5)

                ublox_data = load_gnss_data(data_root, entry_data)
                if not origin_ublox_data:
                    origin_ublox_data = ublox_data

                pcd_points_in_ned_frame, origin_ublox_data = points_to_world_coord_ned(pcd_points,
                                                                                       ublox_data,
                                                                                       lidar2gnss,
                                                                                       origin_ublox_data=origin_ublox_data)

                # Create Open3D point cloud
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(pcd_points_in_ned_frame[:, :3])
                pcd.colors = o3d.utility.Vector3dVector([[0, 0, 1]] * len(pcd.points))  # Blue
                point_clouds_ned.append(pcd)

            if radar_vis:
                radar_path = os.path.join(data_root, entry_data.get('path_to_radar'))
                pcd_points = load_radar_as_pcd(radar_path, intensity_threshold=radar_threshold, image_fov_only=front_fov_only)

                radar2gnss = calib_data["extrinsics"]["radar2gnss"]
                if args.motion_compensation:
                    pcd_points = motion_compensate_pcd(data_root, entry_data, pcd_points,
                                                                   radar2gnss, ts_channel_num=4)

                ublox_data = load_gnss_data(data_root, entry_data)
                if not origin_ublox_data:
                    origin_ublox_data = ublox_data

                pcd_points_in_ned_frame, origin_ublox_data = points_to_world_coord_ned(pcd_points,
                                                                                       ublox_data,
                                                                                       radar2gnss,
                                                                                       origin_ublox_data=origin_ublox_data)

                # Create Open3D point cloud
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(pcd_points_in_ned_frame[:, :3])
                pcd.colors = o3d.utility.Vector3dVector([[1, 0, 0]] * len(pcd.points))  # Red
                point_clouds_ned.append(pcd)

            # Visualize all point clouds
            o3d.visualization.draw_geometries(point_clouds_ned)


if __name__ == "__main__":
    # Argument parser with description
    parser = argparse.ArgumentParser(description='3D data visualization tool of motion compensated point clouds in NED frame')
    parser.add_argument('--muses_root', default='data/muses/', help='Root path for data')
    parser.add_argument('--lidar', action='store_true', help='Activate lidar visualization')
    parser.add_argument('--radar', action='store_true', help='Activate radar visualization')
    parser.add_argument('--radar_threshold', type=int, default=75,
                        help='Intensity threshold for the radar')
    parser.add_argument('--full_radar_fov', action='store_true', default=False,
                        help='Display full 360 degree radar field of view.'
                             'If not selected, only the points in the cameras field of view is displayed.')
    parser.add_argument('--motion_compensation', action='store_true',
                        help='Enable motion compensation  (default: False)')
    parser.add_argument('--split', default=None,
                        help='Selected dataset split: train, val or test')
    parser.add_argument('--time_of_day', default=None,
                        help='Filter dataset by time of day: day or night')
    parser.add_argument('--weather', default=None,
                        help='Filter dataset by condition: clear, rain, snow or fog')

    args = parser.parse_args()

    if args.radar:
        print(f'Visualizing radar data with intensity threshold {args.radar_threshold}')
        if not os.path.exists(os.path.join(args.muses_root, 'radar')):
            raise FileNotFoundError(f'[ERROR] Radar data not found in {os.path.join(args.muses_root, "radar")}.'
                                    f'Please download the data from https://muses.vision.ee.ethz.ch/download '
                                    f'and unzip the sensors data before proceeding.')
    if args.lidar:
        print('Visualizing lidar data')
        if not os.path.exists(os.path.join(args.muses_root, 'lidar')):
            raise FileNotFoundError(f'[ERROR] Lidar data not found in {os.path.join(args.muses_root, "lidar")}.'
                                    f'Please download the data from https://muses.vision.ee.ethz.ch/download '
                                    f'and unzip the sensors data before proceeding.')

    # display message and abort if neither lidar or radar is selected
    if not args.lidar and not args.radar:
        print('[ERROR] Please select lidar or radar for visualization by adding [--lidar] or [--radar] or both.')
        exit(1)

    visualize_3d(args)