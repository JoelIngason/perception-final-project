# 34759 Perception for Autonomous Systems - Final Project (Raw Data)

This README provides information about the raw data provided for the project, including its structure, data formats, and relevant descriptions to facilitate effective utilization and processing.

## Table of Contents

- [Project Overview](#project-overview)
- [Data Contents](#data-contents)
- [Data Format](#data-format)
- [Labels Description](#labels-description)
- [Calibration](#calibration)
- [Important Notes](#important-notes)
- [Usage](#usage)
- [License](#license)
- [Contact](#contact)

## Project Overview

This project focuses on the perception component of autonomous driving systems, aiming to detect, track, and classify objects such as pedestrians, cyclists, and cars within dynamic and unstructured environments. The provided raw data serves as the foundation for developing and evaluating algorithms that ensure safe and reliable autonomous navigation.

## Data Contents

Within the `34759_final_project_raw` folder, you will find the following:

- **Three Recorded Sequences**: Each sequence contains data of pedestrians, cyclists, and cars.
- **Camera Calibration Sequence**: A dedicated sequence for calibrating the stereo camera system.

### Directory Structure

```
34759_final_project_raw/
├── sequences/
│   ├── seq_1/
│   ├── seq_2/
│   └── seq_3/
├── calibration/
│   ├── calib_cam_to_cam.txt
│   └── calibration_patterns/
│       └── pattern_image.png
└── labels/
    ├── labels_seq_1.txt
    ├── labels_seq_2.txt
    └── labels_seq_3.txt  # Note: Labels for seq_3 are not provided
```

## Data Format

Each **sequence folder** (`seq_<n>/`) follows the structure below:

```
seq_<n>/
├── image02/
│   ├── data/
│   │   └── <image_seq_no>.png
│   └── timestamps.txt
└── image03/
    ├── data/
    │   └── <image_seq_no>.png
    └── timestamps.txt
```

- **image02/data/**: Contains the left color camera sequence images in PNG format.
- **image02/timestamps.txt**: Lists the timestamps for each left image. Each line corresponds to the respective image sequence number.
- **image03/data/**: Contains the right color camera sequence images in PNG format.
- **image03/timestamps.txt**: Lists the timestamps for each right image. Each line corresponds to the respective image sequence number.

### Detailed Description

- **Left and Right Images**: 
  - `image02/data/` stores images from the left camera.
  - `image03/data/` stores images from the right camera.
  
- **Timestamps**:
  - `image02/timestamps.txt` and `image03/timestamps.txt` provide synchronization information between image pairs.

- **Labels**:
  - `labels.txt` files are provided for sequences 1 and 2, containing ground truth annotations. **No labels are provided for sequence 3**.

## Labels Description

The label files (`labels_seq_1.txt` and `labels_seq_2.txt`) contain ground truth data for object detection and tracking. Each row in the labels file corresponds to a single object and includes the following **17 columns**:

| Column | Name        | Description                                                                                             |
|--------|-------------|---------------------------------------------------------------------------------------------------------|
| 1      | `frame`     | Frame number within the sequence where the object appears.                                            |
| 2      | `track id`  | Unique tracking ID of the object within the sequence.                                                  |
| 3      | `type`      | Type of object: `Car`, `Pedestrian`, `Cyclist`.                                                        |
| 4      | `truncated` | Float value between 0 (non-truncated) and 1 (truncated), indicating if the object is partially out of view. |
| 5      | `occluded`  | Integer indicating occlusion state: 0 = fully visible, 1 = partly occluded, 2 = largely occluded, 3 = unknown. |
| 6      | `alpha`     | Observation angle of the object, ranging from $[-\pi;\pi]$.                                            |
| 7-10   | `bbox`      | 2D bounding box in rectified image coordinates: left, top, right, bottom pixel coordinates.            |
| 11-13  | `dimensions`| 3D object dimensions in meters: height, width, length.                                                |
| 14-16  | `location`  | 3D object location in camera coordinates (meters): x, y, z.                                          |
| 17     | `rotation_y`| Rotation around the Y-axis in camera coordinates, ranging from $[-\pi;\pi]$.                          |
| 18     | `score`     | (Only for results) Float indicating confidence in detection, used for precision/recall curves. Higher is better. |

**Note**: The 2D bounding boxes are provided with respect to **rectified image coordinates**, even though the data in this folder is raw. Rectification will be performed as part of the data processing pipeline.

## Calibration

Accurate calibration of the stereo camera system is crucial for tasks such as depth estimation and 3D object localization. The calibration data provided includes:

- **calib_cam_to_cam.txt**: Contains camera-to-camera calibration parameters, including intrinsic and extrinsic parameters.
- **calibration_patterns/**: Contains images of calibration patterns used for calibrating the stereo camera system.

### Calibration Parameters

The `calib_cam_to_cam.txt` file includes the following parameters:

- **S_xx**: Size of image xx before rectification (1x2).
- **K_xx**: Calibration matrix of camera xx before rectification (3x3).
- **D_xx**: Distortion coefficients of camera xx before rectification (1x5).
- **R_xx**: Rotation matrix of camera xx (extrinsic, 3x3).
- **T_xx**: Translation vector of camera xx (extrinsic, 3x1).
- **S_rect_xx**: Size of image xx after rectification (1x2).
- **R_rect_xx**: Rectifying rotation matrix to make image planes co-planar (3x3).
- **P_rect_xx**: Projection matrix after rectification (3x4).

These parameters will be utilized in the calibration and rectification stages of the perception pipeline to ensure accurate spatial measurements and object localization.

## Important Notes

- **Label Availability**: Labels are provided only for sequences 1 and 2. Sequence 3 does not include labels and is intended for testing and evaluation purposes without ground truth.
  
- **2D Bounding Boxes**: All 2D bounding boxes in the labels are defined with respect to **rectified image coordinates**. 

- **Data Synchronization**: Timestamps in `timestamps.txt` files are critical for synchronizing left and right images. 