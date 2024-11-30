import argparse
import logging
import logging.config

import cv2
import numpy as np
import torch
import yaml

from src.calibration.stereo_calibration import StereoCalibrator
from src.data_loader.dataset import DataLoader
from src.depth_estimation.depth_estimator import DepthEstimator
from src.models.feature_extractor.feature_extractor import FeatureExtractor
from src.models.model import load_model
from src.tracking.byte_tracker import BYTETracker
from src.visualization.visualizer import Visualizer


def setup_logger(name: str, config_file: str | None = None) -> logging.Logger:
    """Set up logging based on the provided configuration."""
    if config_file:
        with open(config_file) as f:
            config = yaml.safe_load(f)
            logging.config.dictConfig(config)
    logger = logging.getLogger(name)
    return logger


@torch.no_grad()
def run(config_path: str) -> None:
    # Load configuration
    with open(config_path) as file:
        config = yaml.safe_load(file)

    # Setup logger
    logger = setup_logger("autonomous_perception", config.get("logging", {}).get("config_file", ""))
    logger.info(f"Configuration loaded from {config_path}")

    # Initialize calibrator
    use_rectified_data = config["processing"].get("use_rectified_data", True)
    calibrator = StereoCalibrator(config["calibration"])
    calibration_file_key = "calibration_file_rect" if use_rectified_data else "calibration_file_raw"
    calibration_file = config["calibration"].get(calibration_file_key)

    calib_params = calibrator.calibrate(calibration_file)

    depth_estimator = DepthEstimator(calib_params, config)

    # Initialize components
    data_loader = DataLoader(config["data"])
    # Initialize YOLO model with tracking
    yolo = load_model(config["models"]["classification"]["path"])
    # Load data
    data = (
        data_loader.load_sequences_rect(calib_params)
        if use_rectified_data
        else data_loader.load_sequences_raw(calib_params)
    )
    # Retrieve class names from the model
    names = yolo.names  # Ensure the model has a 'names' attribute

    # Initialize FeatureExtractor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    feature_extractor = FeatureExtractor(device=device)

    tracker = BYTETracker(frame_rate=10, feature_extractor=feature_extractor)

    # Initialize Visualizer
    visualizer = Visualizer()

    # Make OpenCV window resizable
    cv2.namedWindow("Autonomous Perception", cv2.WINDOW_NORMAL)

    for seq_id, frames in data.items():
        logger.info(f"Processing sequence {seq_id} with {len(frames)} frames.")
        tracker.reset()

        for frame_idx, frame in enumerate(frames, 1):
            # Perform tracking
            img_left = frame.image_left
            img_right = frame.image_right

            # Compute disparity and depth maps
            disparity_map = depth_estimator.compute_disparity_map(img_left, img_right)
            depth_map = depth_estimator.compute_depth_map(disparity_map)

            # Perform object detection
            results = yolo(img_left)[0].boxes.cpu().numpy()
            logger.debug(f"Frame {frame_idx}: {len(results)} detections.")

            # Add zs to detection results
            # Assuming results have xywh format: [x_center, y_center, width, height]
            # and we need to insert z (depth)
            detections = []
            measurement_masks = []
            for i, result in enumerate(results):
                bbox = result.xywh[0].tolist()  # [x_center, y_center, width, height]
                depth, object_depth_range = depth_estimator.compute_object_depth(bbox, depth_map)
                if (
                    depth is not None and not np.isnan(depth) and 0 < depth < 6
                ):  # Example valid depth range
                    # Valid depth
                    detection = [bbox[0], bbox[1], depth, bbox[2], bbox[3]]
                    mask = [True, True, True, True, True]
                else:
                    # Missing or invalid depth
                    detection = [bbox[0], bbox[1], 0.0, bbox[2], bbox[3]]
                    mask = [True, True, False, True, True]
                detections.append(detection)
                measurement_masks.append(mask)

            detections = np.array(detections, dtype=np.float32)
            measurement_masks = np.array(measurement_masks, dtype=bool)

            # Create additional columns for scores and class labels if not already present
            # Assuming 'result' has 'conf' and 'cls' attributes
            if len(results) > 0 and hasattr(results[0], "conf") and hasattr(results[0], "cls"):
                scores = np.array([res.conf for res in results], dtype=np.float32)
                cls_labels = np.array([res.cls for res in results], dtype=np.int32)
                # Ensure that detections and measurement_masks are filtered based on valid_depth
                scores = scores[: len(detections)]
                cls_labels = cls_labels[: len(detections)]
            else:
                # Default scores and class labels
                scores = np.ones(len(detections), dtype=np.float32)
                cls_labels = np.zeros(len(detections), dtype=np.int32)

            # Reshape scores and cls_labels to 2D arrays
            scores = scores.reshape(-1, 1)  # Shape: (N, 1)
            cls_labels = cls_labels.reshape(-1, 1)  # Shape: (N, 1)

            # Concatenate detections with scores and class labels
            detections = np.hstack((detections, scores, cls_labels))

            # Update tracker with detections and measurement masks
            _ = tracker.update(detections, measurement_masks, img_left)

            tracks_lost = tracker.lost_stracks
            tracks_active = tracker.tracked_stracks

            logger.debug(
                f"Frame {frame_idx}: {len(tracks_active)} active tracks, {len(tracks_lost)} lost tracks.",
            )

            # Annotate frame with tracks using Visualizer
            annotated_frame = visualizer.display(
                img_left=img_left,
                img_right=img_right,
                tracks_active=tracks_active,
                tracks_lost=tracks_lost,
                names=names,
                depth_map=depth_map,
                conf=True,
                line_width=2,
                font_size=0.5,
                font=cv2.FONT_HERSHEY_SIMPLEX,
                labels=True,
                show=True,  # Set to True to display the annotated image
                save=False,  # Set to True to save the annotated image
                filename=None,  # Specify a filename if saving is enabled
                color_mode="class",
            )

            # Optionally, handle the return value if needed
            if annotated_frame is False:
                # Visualization requested exit
                return

    # Release resources and close windows
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enhanced Autonomous Perception Pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to the configuration YAML file",
    )
    args = parser.parse_args()
    run(args.config)
