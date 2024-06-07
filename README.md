# MUSES: The Multi-Sensor Semantic Perception Dataset for Driving under Uncertainty 

**[[Website]](https://muses.vision.ee.ethz.ch)**
**[[Paper]](https://arxiv.org/pdf/2401.12761.pdf)**

Welcome to the Software Development Kit (SDK) for the MUESS. 
This SDK is designed to help you navigate and manipulate the multiple modalities of the MUSES dataset effectively.

### Overview

![MUSES Overview Figure](resources/paper-diagram-car-vsm.png)

MUSES, the MUlti-SEnsor Semantic perception dataset for driving under uncertainty is a multi-modal adverse condition dataset focused on 2D dense semantic perception.

It is composed of 2500 multi-modal scenes, each of which includes data from a frame camera, a lidar, a radar, an event camera, and an IMU/GNSS sensor. Each scene is accompanied by high-quality pixel-level panoptic annotations for the frame camera. The annotation process for MUSES utilizes all available sensor data, allowing the annotators to also reliably label degraded image regions that are still discernible in other modalities. This results in better pixel coverage in the annotations and creates a more challenging evaluation setup.
 
Further, each adverse-condition image comes with a corresponding image of the same scene taken under clear-weather, daytime conditions.
 
The dataset supports:
1. Panoptic segmentation
2. Uncertainty-aware panoptic segmentation
3. Semantic segmentation
4. Object detection

The ground-truth annotations of the training and validation sets are publicly available. The ground-truth annotations of the test set are withheld and they are used to establish the publicly available [MUSES benchmark].

### Download

MUSES can be downloaded through the [associated website](https://muses.vision.ee.ethz.ch/download). Users need to register on this website first before accessing the dataset.

Unzip all packages into the same folder:
```
unzip /path/to/downloaded_file.zip -d /path/to
```

You should get the following structure:
```
/path/to/muses/
├── frame_camera
├── gt_panoptic
├── gt_uncertainty
├── gt_semantic
├── gt_detection
├── lidar
├── radar
├── event_camera
├── reference_frames
├── gnss
├── calib.json
├── meta.json
├── LICENSE.pdf
└── README.md
```

To work with a subset of the dataset, (i.e. standard panoptic segmentation on RGB) you can download only the relevant folders:
```
/path/to/muses/
├── gt_panoptic
├── frame_camera
├── meta.json
├── LICENSE.pdf
└── README.md
```

**Note**: Due to data protection regulations, only anonymized versions of the images are available.

For ease of use, you can link the MUSES dataset folder to `./data/muses` (assumed as default in the scripts) by running:

```bash
mkdir -p data
ln -s /path/to/muses ./data/muses
```

### Installation

The SDK requires Python 3.6 or higher. To install the required packages, run:

```bash
conda create -n muses python=3.11
conda activate muses
pip install -e .
```

### SDK

In this SDK, we provide the following tools:

1. [Visualization](visualization): Scripts to visualize the dataset.
   - [visualize.py](visualization/visualization.py): A script to visually inspect the dataset in 2D. Including projected overlays of all add additional modalities.
   - [visualization_3d.py](visualization/visualization_3d.py): A script to visually inspect individual scenes in 3D.

2. [AUPQ](AUPQ): Script to compute the AUPQ metric.
    - [uncertainty_aware_panoptic_quality.py](AUPQ/uncertainty_aware_panoptic_quality.py): A script to compute our AUPQ metric.

3. [scripts](scripts): Scripts to process the dataset.
   - [project_sensors_to_rgb.py](scripts/project_sensors_to_rgb.py): A script to project the sensor data to the frame camera.
   - [calculate_channel_statistics.py](scripts/calculate_channel_statistics.py): A script to calculate channel statistics for the projected sensor images.
   - [motion_compensation.py](scripts/motion_compensation.py): A script to perform motion compensation for the lidar and radar data.

The SDK is further ment to give you a starting point for your own scripts. 
I.e. in the [processing](processing) folder, you can find sensor specific helper functions for lidar, radar and event camera to load and process the data,
like 'load_event_camera_data', 'load_lidar_data' or 'load_radar_data'. 
In the [utils](utils) folder, you can find further helper functions like 'load_muses_calibration_data'or 'load_gnss_data'.

### Citation

If you use MUSES in your work, please cite our publications as listed on the [MUSES website](https://muses.vision.ee.ethz.ch/citation).

### License

MUSES is made available for non-commercial use under the license agreement which is contained in the attached file `License.pdf`.

### Acknowledgements

This work was supported by the ETH Future Computing Laboratory (EFCL), financed by a donation from Huawei Technologies.

### Contact

Please feel free to contact us with any questions or comments:

Tim Broedermann, David Bruggemann and Christos Sakaridis

E-Mail: muses.efcl [at] zohomail.eu

Website: https://muses.vision.ee.ethz.ch

