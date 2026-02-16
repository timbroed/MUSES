# MUSES: The Multi-Sensor Semantic Perception Dataset for Driving under Uncertainty 

**[[Download]](https://muses.ethz.ch)**
**[[Paper]](https://arxiv.org/pdf/2401.12761.pdf)**

**Benchmarks:**
- **[Panoptic Segmentation](https://www.codabench.org/competitions/13987/)** 
- **[Semantic Segmentation](https://www.codabench.org/competitions/14005/)** 

:bell: **News:**
* [2026-02-16] We host the MUSES benchmarks on codabench.
* [2025-12-31] The orignal website (muses.vision.ee.ethz.ch) was deactivated. You can download the dataset now under the [new website link](https://muses.ethz.ch).
* [2025-04-11] [Cafuser](https://github.com/timbroed/CAFuser) was released, including a Detectron2 [DataLoader](https://github.com/timbroed/CAFuser/blob/master/cafuser/data/dataset_mappers/muses_unified_dataset_mapper.py) for MUSES.
* [2024-07-17] We are happy to announce that MUSES was accepted at **ECCV 2024**.

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

The ground-truth annotations of the training and validation sets are publicly available. The ground-truth annotations of the test set are withheld and they are used to establish the publicly available [Panoptic Segmentation benchmark](https://www.codabench.org/competitions/13987/) and [Semantic Segmentation benchmark](https://www.codabench.org/competitions/14005/).

### Download

MUSES can be downloaded through the [associated website](https://muses.ethz.ch).

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

The SDK is further meant to give you a starting point for your scripts. 
I.e. in the [processing](processing) folder, you can find sensor-specific processing files for [lidar](processing/lidar_processing.py), [radar](processing/radar_processing.py) and [event camera](processing/event_camera_processing.py) to load and process the modality.
For loading the unprocessed sensor data you can use: 'load_lidar_data', 'load_radar_as_pcd', and 'accumulate_events'. 
For loading the modalities projected onto the camera image you can use: 'load_lidar_projection', 'load_radar_projection', and 'load_event_camera_projection'.
In the [utils](processing/utils.py) file, you can find further helper functions like 'load_meta_data', 'load_muses_calibration_data', and 'read_gnss_file'.

### DataLoader

You can find an exemplary Detectron2 [DataLoader](https://github.com/timbroed/CAFuser/blob/master/cafuser/data/dataset_mappers/muses_unified_dataset_mapper.py) for MUSES in the [Cafuser repository](https://github.com/timbroed/CAFuser).

### Citation

If you use MUSES in your work, please cite our publications:

```bib
@inproceedings{brodermann2024muses,
  title={MUSES: The Multi-Sensor Semantic Perception Dataset for Driving under Uncertainty},
  author={Br{\"o}dermann, Tim and Bruggemann, David and Sakaridis, Christos and Ta, Kevin and Liagouris, Odysseas and Corkill, Jason and Van Gool, Luc},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2024}
}
```

### License

MUSES is made available for non-commercial use under the license agreement which is contained in the attached file `License.pdf`.

### Acknowledgements

This work was supported by the ETH Future Computing Laboratory (EFCL), financed by a donation from Huawei Technologies.

### Contact

Please feel free to contact us with any questions or comments:

Tim Broedermann and Christos Sakaridis

E-Mail: muses.efcl [at] zohomail.eu

Website: https://muses.vision.ee.ethz.ch
