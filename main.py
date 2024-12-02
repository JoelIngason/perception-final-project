import argparse
import logging
import logging.config
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from attr import has

from src.calibration.stereo_calibration import StereoCalibrator
from src.data_loader.dataset import DataLoader, TrackedObject
from src.depth_estimation.depth_estimator import DepthEstimator
from src.evaluation.evaluator import Evaluator
from src.models.feature_extractor.feature_extractor import FeatureExtractor
from src.models.model import load_model
from src.tracking.byte_tracker import BYTETracker
from src.utils.label_parser import parse_labels
from src.visualization.visualizer import Visualizer


def is_light_color(color):
    """
    Determine if a BGR color is light based on its luminance.

    Args:
        color (tuple): BGR color tuple.

    Returns:
        bool: True if the color is light, False otherwise.
    """
    luminance = 0.299 * color[2] + 0.587 * color[1] + 0.114 * color[0]
    return luminance > 186  # Threshold for light colors


def setup_logger(name: str, config_file: str | None = None) -> logging.Logger:
    """Set up logging based on the provided configuration."""
    if config_file and Path(config_file).is_file():
        with open(config_file) as f:
            config = yaml.safe_load(f)
            logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(name)
    return logger


@torch.no_grad()
def run(config_path: str) -> None:
    # Load configuration
    with open(config_path) as file:
        config = yaml.safe_load(file)

    # Setup logger
    logger = setup_logger(
        "autonomous_perception",
        config.get("logging", {}).get("config_file", None),
    )
    logger.info(f"Configuration loaded from {config_path}")

    # Initialize Evaluator
    evaluator = Evaluator()

    # Initialize calibrator
    use_rectified_data = config["processing"].get("use_rectified_data", True)
    calibrator = StereoCalibrator(config["calibration"])
    calibration_file_key = "calibration_file_rect" if use_rectified_data else "calibration_file_raw"
    calibration_file = config["calibration"].get(calibration_file_key)

    calib_params = calibrator.calibrate(calibration_file)
    logger.info(f"Calibration parameters loaded: {calib_params}")

    depth_estimator = DepthEstimator(calib_params, config)

    # Initialize components
    data_loader = DataLoader(config["data"])

    # Initialize YOLO model with tracking
    yolo = load_model(config["models"]["classification"]["path"])
    if not hasattr(yolo, "names"):
        logger.error("Loaded YOLO model does not have a 'names' attribute.")
        return
    names = yolo.names  # Ensure the model has a 'names' attribute
    logger.info(f"YOLO model loaded with classes: {names}")

    # Load data
    data = (
        data_loader.load_sequences_rect(calib_params)
        if use_rectified_data
        else data_loader.load_sequences_raw(calib_params)
    )
    logger.info(f"Loaded {len(data)} sequences.")

    # Initialize FeatureExtractor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    feature_extractor = FeatureExtractor(device=device)
    logger.info(f"FeatureExtractor initialized on device: {device}")

    tracker_pedastrians = BYTETracker(
        "config/byte_tracker_pedestrians.yaml",
        frame_rate=config["processing"].get("frame_rate", 10),
        feature_extractor=feature_extractor,
    )
    tracker_cars = BYTETracker(
        "config/byte_tracker_cars.yaml",
        frame_rate=config["processing"].get("frame_rate", 10),
        feature_extractor=feature_extractor,
    )
    tracker_cyclists = BYTETracker(
        "config/byte_tracker_cyclists.yaml",
        frame_rate=config["processing"].get("frame_rate", 10),
        feature_extractor=feature_extractor,
    )

    logger.info("BYTETracker initialized.")

    # Initialize Visualizer
    visualizer = Visualizer()

    # Make OpenCV window resizable
    cv2.namedWindow("Autonomous Perception", cv2.WINDOW_NORMAL)

    # Iterate over each sequence
    for seq_id, frames in data.items():
        logger.info(f"Processing sequence {seq_id} with {len(frames)} frames.")
        tracker_pedastrians.reset()
        tracker_cars.reset()
        tracker_cyclists.reset()
        # Load ground truth labels
        label_files = config["data"].get("label_files", "")
        if not label_files:
            logger.error(
                "Label file path not specified in the configuration under 'data.label_file'."
            )
            return

        has_label = True
        try:
            ground_truth_labels = parse_labels(label_files[int(seq_id)])
        except KeyError:
            has_label = False
            ground_truth_labels = {}

        for frame_idx, frame in enumerate(frames, 1):
            # Perform tracking
            img_left = frame.image_left
            img_right = frame.image_right

            # Compute disparity and depth maps
            disparity_map = depth_estimator.compute_disparity_map(img_left, img_right)
            depth_map = depth_estimator.compute_depth_map(disparity_map, img_left)
            logger.debug(f"Frame {frame_idx}: Disparity and depth maps computed.")

            # Perform object detection
            results = yolo(img_left)[0].boxes.cpu().numpy()
            logger.debug(f"Frame {frame_idx}: {len(results)} detections.")

            # Process detections to include depth (z)
            detections = []
            measurement_masks = []
            centroids = []
            for result in results:
                bbox = result.xywh[0].tolist()  # [x_center, y_center, width, height]
                depth, centroid = depth_estimator.compute_object_depth(bbox, depth_map)
                centroids.append(centroid)
                if (
                    depth is not None and not np.isnan(depth) and 0 < depth < 6
                ):  # Example valid depth range
                    detection = [bbox[0], bbox[1], depth, bbox[2], bbox[3]]
                    mask = [True, True, True, True, True]
                else:
                    detection = [bbox[0], bbox[1], 0.0, bbox[2], bbox[3]]
                    mask = [True, True, False, True, True]
                detections.append(detection)
                measurement_masks.append(mask)

            detections = np.array(detections, dtype=np.float32)
            measurement_masks = np.array(measurement_masks, dtype=bool)

            # Extract scores and class labels
            if len(results) > 0 and hasattr(results[0], "conf") and hasattr(results[0], "cls"):
                scores = np.array([res.conf for res in results], dtype=np.float32)
                cls_labels = np.array([res.cls for res in results], dtype=np.int32)
            else:
                scores = np.ones(len(detections), dtype=np.float32)
                cls_labels = np.zeros(len(detections), dtype=np.int32)

            # Reshape and concatenate scores and class labels
            scores = scores.reshape(-1, 1)  # Shape: (N, 1)
            cls_labels = cls_labels.reshape(-1, 1)  # Shape: (N, 1)
            detections = np.hstack((detections, scores, cls_labels))

            # Update trackers with detections tracker with detections and measurement masks
            # active_tracks = tracker.update(detections, measurement_masks, img_left)
            detections_pedastrians = detections[cls_labels.flatten() == 0]
            detections_cars = detections[cls_labels.flatten() == 2]
            detections_cyclists = detections[cls_labels.flatten() == 1]

            measurement_masks_pedastrians = measurement_masks[cls_labels.flatten() == 0]
            measurement_masks_cars = measurement_masks[cls_labels.flatten() == 2]
            measurement_masks_cyclists = measurement_masks[cls_labels.flatten() == 1]

            active_tracks_pedastrians = tracker_pedastrians.update(
                detections_pedastrians,
                measurement_masks_pedastrians,
                img_left,
            )

            active_tracks_cars = tracker_cars.update(
                detections_cars,
                measurement_masks_cars,
                img_left,
            )

            active_tracks_cyclists = tracker_cyclists.update(
                detections_cyclists,
                measurement_masks_cyclists,
                img_left,
            )

            non_empty_tracks = [
                tracks
                for tracks in (
                    active_tracks_pedastrians,
                    active_tracks_cars,
                    active_tracks_cyclists,
                )
                if tracks.size > 0
            ]
            active_tracks = (
                np.vstack(non_empty_tracks) if non_empty_tracks else np.empty((0, 5))
            )  # Adjust the shape as needed
            logger.debug(f"Frame {frame_idx}: {len(active_tracks)} active tracks.")

            tracked_tracks = (
                tracker_pedastrians.tracked_stracks
                + tracker_cars.tracked_stracks
                + tracker_cyclists.tracked_stracks
            )
            lost_tracks = (
                tracker_pedastrians.lost_stracks
                + tracker_cars.lost_stracks
                + tracker_cyclists.lost_stracks
            )

            all_tracks = tracked_tracks + lost_tracks

            # Convert active tracks to TrackedObject instances for evaluation
            tracked_objects = []
            for track in all_tracks:
                # Extract position_3d as center coordinates
                x1, y1, z, x2, y2 = track.xyzxy

                # Estimate dimensions
                width = x2 - x1
                height = y2 - y1
                length = 2.0  # Example value

                # Compute position_3d as centroid
                position_3d = [x1 + width / 2, y1 + height / 2, z]

                # Compute dimensions
                dimensions = [height, width, length]

                tracked_obj = TrackedObject(
                    track_id=int(track.track_id),
                    label=names[int(track.cls)],
                    bbox=[x1, y1, x2, y2],
                )
                tracked_objects.append(tracked_obj)

            if has_label:
                # Retrieve ground truth objects for the current frame
                gt_objects = ground_truth_labels.get(frame_idx, [])

                # Evaluate the current frame
                evaluator.evaluate(tracked_objects, gt_objects)

                # Create a copy of img_left and draw ground truth boxes
                img_left_gt = img_left.copy()

                for gt_obj in gt_objects:
                    bbox = gt_obj.bbox  # [left, top, right, bottom]
                    cls_name = gt_obj.obj_type

                    # Define color for ground truth (e.g., Yellow)
                    gt_color = (0, 255, 255)  # Yellow in BGR

                    # Draw bounding box
                    cv2.rectangle(
                        img_left_gt,
                        (int(bbox[0]), int(bbox[1])),
                        (int(bbox[2]), int(bbox[3])),
                        gt_color,
                        2,
                    )

                    cls = cls_name
                    depth = gt_obj.location[2]
                    # Define label text
                    label = f"GT: {cls_name} - Depth: {depth:.2f}m"

                    # Calculate text size for background rectangle
                    (text_width, text_height), baseline = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                    )

                    # Draw background rectangle for text
                    cv2.rectangle(
                        img_left_gt,
                        (int(bbox[0]), int(bbox[1]) - text_height - baseline),
                        (int(bbox[0]) + text_width, int(bbox[1])),
                        gt_color,
                        thickness=cv2.FILLED,
                    )

                    # Choose text color based on background brightness for readability
                    text_color = (0, 0, 0) if is_light_color(gt_color) else (255, 255, 255)

                    # Put text label
                    cv2.putText(
                        img_left_gt,
                        label,
                        (int(bbox[0]), int(bbox[1]) - baseline),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        text_color,
                        1,
                        cv2.LINE_AA,
                    )
            else:
                # If no ground truth, create an empty image or duplicate img_left
                img_left_gt = img_left.copy()

            # Annotate frame with tracks using Visualizer
            annotated_frame = visualizer.display(
                img_left=img_left,
                img_right=img_left_gt,  # Pass the annotated img_left as img_right
                tracks_active=tracked_tracks,
                tracks_lost=lost_tracks,
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
                centroids=centroids,
            )

            # Optionally, handle the return value if needed
            if annotated_frame is False:
                # Visualization requested exit
                logger.info("Visualization requested exit. Terminating run.")
                return

        # After processing all frames, generate evaluation report
        evaluator.report()

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
