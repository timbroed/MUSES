# 2D and 3D Visualization

This sub-package includes two main scripts: visualization.py and visualization_3d.py. 
These scripts are designed to visualize sensor data, including LiDAR, radar, and event camera data.  

### visualization.py

A script to visually inspect the dataset in 2D. Including projected overlays of all add additional modalities.
You can navigate through the dataset by pressing the left and right arrow keys and switch between different modalities by pressing the corresponding number key.

```
python visualization/visualization.py --muses_root <muses_root> --motion_compensation --split <split> --time_of_day <time_of_day> --weather <weather> --image <image> --canvas_scale <canvas_scale> --radar_threshold <radar_threshold>
```

- `--muses_root`: (optional) The root path for the muses dataset. Default is `data/muses/`.
- `--motion_compensation`: (optional) If present, the script will enable motion compensation.
- `--split`: (optional) The selected dataset split: train, val, or test.
- `--time_of_day`: (optional) Filter the dataset by time of day: day or night.
- `--weather`: (optional) Filter the dataset by condition: clear, rain, snow, or fog.
- `--image`: (optional) The number of the starting image.
- `--canvas_scale`: (optional) The scale factor for the program canvas. Change to adapt the size to your screen.
- `--radar_threshold`: (optional) Intensity threshold for the radar (Default: 75). Set to 0 to see the full radar image.

#### Example

To view all scenes in the dataset with motion compensation and enlarged points for better visibility, run the following command:
```
python visualization/visualization.py --muses_root data/muses/
```

### visualization_3d.py

A script to visually inspect individual scenes in 3D.
The script will simply loop over all scenes in the selected split and display them in an interactive 3D viewer.
To move to the next scenes, close the current window. 
Radar is displayed in red and lidar in blue.

```
python visualization/visualization_3d.py --muses_root <muses_root> --lidar --radar --radar_threshold <radar_threshold> --full_radar_fov --motion_compensation --split <split> --time_of_day <time_of_day> --weather <weather>
```

- `--muses_root`: (optional) The root path for the msues dataset. Default is `data/muses/`.
- `--lidar`: (optional) Activate lidar visualization.
- `--radar`: (optional) Activate radar visualization.
- `--radar_threshold`: (optional) Intensity threshold for the radar.
- `--full_radar_fov`: (optional) If present, the script will display the full radar field of view. Default is to only display the radar points in the camera field of view.
- `--motion_compensation`: (optional) If present, the script will enable motion compensation.
- `--split`: (optional) The selected dataset split: train, val, or test.
- `--time_of_day`: (optional) Filter the dataset by time of day: day or night.
- `--weather`: (optional) Filter the dataset by condition: clear, rain, snow, or fog.

#### Example

To visualize all scenes in the dataset with LiDAR and radar data in 3D with motion compensation, run the following command:
```
python visualization/visualization_3d.py --muses_root data/muses/ --lidar --radar --motion_compensation 
```