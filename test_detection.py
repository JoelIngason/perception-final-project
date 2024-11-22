import argparse
import logging
from pathlib import Path

import numpy as np
import yaml

# Importing the necessary modules from your project structure
from src.calibration.stereo_calibration import StereoCalibrator
from src.data_loader.dataset import DataLoader, TrackedObject
from src.depth_estimation.depth_estimator import DepthEstimator
from src.detection.detector import ObjectDetector
from src.evaluation.evaluator import Evaluator
from src.utils.label_parser import parse_labels
from src.visualization.visualizer import Visualizer


def setup_logger(name: str, config_file: str | None = None) -> logging.Logger:
    """Set up logging based on the provided configuration."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)  # Set to DEBUG to capture all logs

    # Prevent adding multiple handlers if they already exist
    if not logger.hasHandlers():
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def load_config(config_path: str) -> dict:
    """Load the YAML configuration file."""
    with open(config_path) as file:
        config = yaml.safe_load(file)
    return config


def main(config_path: str, sequence_id: str, frame_number: int) -> None:
    """Execute the detection and evaluation on a specific frame."""
    # Load configuration
    config = load_config(config_path)

    # Setup logger
    logger = setup_logger("test_detection")
    logger.info(f"Configuration loaded from {config_path}")

    # Initialize calibrator
    calibrator = StereoCalibrator(config["calibration"])
    calibration_file_rect = config["calibration"].get("calibration_file_rect")
    if not calibration_file_rect or not Path(calibration_file_rect).exists():
        logger.error(f"Calibration file '{calibration_file_rect}' does not exist.")
        return

    calib_params = calibrator.calibrate(calibration_file_rect)
    logger.info("Calibration parameters loaded and calibrated.")

    # Initialize DataLoader
    data_loader = DataLoader(config["data"])

    # Decide whether to use rectified data
    use_rectified = config["processing"].get("use_rectified_data", True)
    if use_rectified:
        sequences = data_loader.load_sequences_rect(calib_params)
        logger.info("Loaded rectified sequences.")
    else:
        sequences = data_loader.load_sequences_raw(calib_params)
        logger.info("Loaded raw sequences.")

    if not sequences:
        logger.error("No sequences loaded. Exiting test.")
        return

    # Select the specified sequence
    if sequence_id not in sequences:
        logger.error(f"Sequence ID '{sequence_id}' not found in loaded sequences.")
        return

    frames = sequences[sequence_id]
    if frame_number < 0 or frame_number >= len(frames):
        logger.error(f"Frame number '{frame_number}' is out of range for sequence '{sequence_id}'.")
        return

    # Select the specified frame
    frame = frames[frame_number]
    logger.info(f"Selected Sequence '{sequence_id}', Frame '{frame_number}'.")

    # Initialize ObjectDetector
    detector = ObjectDetector(config, calib_params)

    # Initialize DepthEstimator
    depth_estimator = DepthEstimator(calib_params, config)

    # Initialize Evaluator
    evaluator = Evaluator()

    # Initialize Visualizer
    visualizer = Visualizer()

    # Perform detection on left and right images
    logger.info("Performing object detection on left image.")
    detections_left = detector.detect(frame.image_left, "left")
    logger.info(f"Detections Left: {len(detections_left)} objects detected.")

    logger.info("Performing object detection on right image.")
    detections_right = detector.detect(frame.image_right, "right")
    logger.info(f"Detections Right: {len(detections_right)} objects detected.")

    # Compute disparity and depth maps
    logger.info("Computing disparity map.")
    disparity_map = depth_estimator.compute_disparity_map(frame.image_left, frame.image_right)
    logger.info("Disparity map computed.")

    logger.info("Computing depth map.")
    depth_map = depth_estimator.compute_depth_map(disparity_map)
    logger.info("Depth map computed.")

    # Update tracked objects with depth information
    # For simplicity, we treat detections as tracked objects
    tracked_objects_left = []
    for det in detections_left:
        # Estimate dimensions using the depth map
        det.estimate_dimensions(depth_map)

        # Convert DetectionResult to TrackedObject
        tracked_obj = TrackedObject(
            track_id=-1,  # -1 indicates that tracking ID is not assigned in this test
            bbox=[int(coord) for coord in det.bbox],  # [left, top, right, bottom]
            confidence=det.confidence,
            class_id=det.class_id,
            label=det.label,  # Assuming 'label' is directly accessible
            position_3d=list(det.position_3d) if det.position_3d else [0.0, 0.0, 0.0],
            dimensions=list(det.dimensions) if det.dimensions else [0.0, 0.0, 0.0],
            mask=det.mask if det.mask.size > 0 else np.array([]),
        )
        tracked_objects_left.append(tracked_obj)

    tracked_objects_right = []
    for det in detections_right:
        # Estimate dimensions using the depth map
        det.estimate_dimensions(depth_map)

        # Convert DetectionResult to TrackedObject
        tracked_obj = TrackedObject(
            track_id=-1,  # -1 indicates that tracking ID is not assigned in this test
            bbox=[int(coord) for coord in det.bbox],  # [left, top, right, bottom]
            confidence=det.confidence,
            class_id=det.class_id,
            label=det.label,  # Assuming 'label' is directly accessible
            position_3d=list(det.position_3d) if det.position_3d else [0.0, 0.0, 0.0],
            dimensions=list(det.dimensions) if det.dimensions else [0.0, 0.0, 0.0],
            mask=det.mask if det.mask.size > 0 else np.array([]),
        )
        tracked_objects_right.append(tracked_obj)

    # Merge tracked objects from both cameras
    tracked_objects = tracked_objects_left + tracked_objects_right
    logger.info(f"Total tracked objects: {len(tracked_objects)}")

    # Load ground truth labels for the current frame
    label_files = config["data"].get("label_files", [])
    if not label_files:
        logger.error("No label files specified in configuration.")
        return

    # Convert sequence_id to integer index
    try:
        sequence_index = int(sequence_id)
    except ValueError:
        logger.error(f"Invalid sequence ID '{sequence_id}'. Must be an integer.")
        return

    # Check if sequence_index is within the label_files list
    if sequence_index >= len(label_files):
        logger.error(f"Sequence ID '{sequence_id}' is out of range for label_files.")
        return

    label_file = label_files[sequence_index]
    if not label_file or not Path(label_file).exists():
        logger.warning(f"No label file found for sequence '{sequence_id}'. Skipping evaluation.")
        gt_objects = []  # No ground truth for this sequence
    else:
        ground_truth_labels = parse_labels(label_file)
        logger.info(f"Parsed {len(ground_truth_labels)} frames from '{label_file}'.")

        # Filter ground truth labels for the current frame
        gt_objects = ground_truth_labels.get(frame_number, [])
        logger.info(f"Ground truth objects for frame {frame_number}: {len(gt_objects)}")

    # Evaluate tracking
    evaluator.evaluate(tracked_objects, gt_objects)
    logger.info("Evaluation completed.")

    # Report metrics
    evaluator.report()

    # Optionally visualize detections and ground truth
    # Uncomment the following lines if you wish to see the visualization
    """
    visualizer.display(
        frame=frame,
        tracked_objects_left=tracked_objects_left,
        tracked_objects_right=tracked_objects_right,
        depth_map=depth_map,
        ground_truth=gt_objects
    )
    """

    logger.info("Test detection completed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Object Detection and Evaluation")
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to the configuration YAML file",
    )
    parser.add_argument(
        "--sequence",
        type=str,
        required=True,
        help="Sequence ID to test (e.g., '1', '2', '3')",
    )
    parser.add_argument(
        "--frame",
        type=int,
        required=True,
        help="Frame number within the sequence to test (starting from 0)",
    )
    args = parser.parse_args()

    main(args.config, args.sequence, args.frame)
