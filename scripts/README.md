### Project sensors to frame camera

To project the sensor data to the frame camera, we provide the following scripts:

```
python scripts/project_sensors_to_rgb.py --muses_root <muses_root> --output_folder <output_folder> --event_camera --lidar --radar --motion_compensation --enlarge_lidar_points --enlarge_radar_points --output_file_type <output_file_type>
```
 
- `--muses_root`: The root directory where the dataset is located. Default is `data/muses/`.
- `--output_folder`: (optional) The name of the output folder where the processed data will be saved. Default is `projected_to_rgb`.
- `--event_camera`: (optional) If present, the script will process event camera data.
- `--lidar`: (optional) If present, the script will process lidar data.
- `--radar`: (optional) If present, the script will process radar data.- `--motion_compensation`: If present, the script will enable motion compensation for lidar and radar data.
- `--enlarge_lidar_points`: (optional) If present, the script will enable points enlargement via dilation for lidar data.
- `--enlarge_radar_points`: (optional) If present, the script will enable points enlargement via dilation for radar data.
- `--output_file_type`: (optional) The file type of the output images. Default is `.npz` to store the images as float32 numpy arrays. The other options are `.npy` and `.png` to store the images as a uint16 PNG file, for this every value (for radar and lidar) gets shifted by 100 and scaled by a factor of 150 to fall within the uint16 range.

The script will save the projected sensor data as `.npz` files or uint16 PNG Images in the output folder.

#### Example

To convert all sensor data to the frame camera, run the following command:

```commandline
python scripts/project_sensors_to_rgb.py --muses_root data/muses/ --event_camera --lidar --radar --output_folder projected_to_rgb --motion_compensation
```

This will save a motion compensated and projected `.png` file for each sensor modality in the output folder `projected_to_rgb`.

### Calculate channel statistics for projected sensor images

To calculate the channel statistics for the projected images, we provide the following script:
```
python scripts/calculate_channel_statistics.py --muses_root <muses_root> --lidar --radar --event_camera --projection_folder <projection_folder> --save_as_json  
```

- `--muses_root`: The root directory where the dataset is located. Default is `data/muses/`.
- `--lidar`: (optional) If present, the script will process lidar data.
- `--radar`: (optional) If present, the script will process radar data.
- `--event_camera`: (optional) If present, the script will process event camera data.
- `--projection_folder`: (optional) The folder containing the projected images. Default is `projected_to_rgb`.
- `--save_as_json`: (optional) If present, the script will save the channel statistics as a JSON file in the `projection_folder`.

The script will print the mean and standard deviation of pixel values for each channel in the images and save these values to a JSON file.
These values can then be used for normalization when training a model on the dataset.

#### Example

To calculate the channel statistics for the projected images of all sensors in the default projections folder, run the following command:

```commandline
python scripts/calculate_channel_statistics.py --muses_root data/muses/ --lidar --radar --event_camera
```

### Motion compensation

To perform motion compensation for the lidar and radar data, we provide the following script:

```
python scripts/motion_compensation.py --dataset_dir <path_to_dataset> --output_dir <path_to_output> --lidar --radar --radar_threshold <radar_threshold> --radar_fov <radar_fov> --output_file_type <output_file_type> --output_folder <output_folder>
```

- `--muses_root`: The path to the directory containing the dataset. Default is `data/muses/`.
- `--output_dir`: (optional) The path to the output directory. If not provided, it will use the dataset directory.
- `--lidar`: (optional) If present, the script will process lidar data.
- `--radar`: (optional) If present, the script will process radar data.
- `--radar_threshold`: (optional) Intensity threshold for the radar. Default is 75. Set to 0 to keep all points.
- `--radar_fov`: (optional) Field of view for the radar visualization with two options: 'all' or 'image'. Default is 'all'. Set to 'image' to only keep the points in the cameras Field of View.
- `--output_file_type`: (optional) The file type of the output point cloud files. Default is `bin` to save the point cloud data as binary `.bin` files. The other options are `npy` and `npz` to save the point cloud data as numpy arrays.
- `--output_folder`: (optional) The name of the output folder where the motion compensated point clouds will be saved. Default is `motion_compensated`.

This script uses the assumption of constant planar motion to compensate for the ego-motion of the vehicle. 
It will save the motion compensated point cloud data as binary `.bin` point cloud files, for both the lidar and the radar.

#### Example

Run the following command, to perform motion compensation for the lidar and radar data, while keeping only the radar points in the camera field of view above a threshold of 75: 

```commandline
python scripts/motion_compensation.py --muses_root ./data/muses  --lidar --radar --radar_threshold 75 --radar_fov image
```
