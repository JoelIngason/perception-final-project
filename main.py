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
from src.evaluation.metrics import (
    compute_mota_motp,
    compute_precision_recall,
    compute_precision_recall_per_class,
)
from src.models.feature_extractor.feature_extractor import FeatureExtractor
from src.models.model import load_model
from src.tracking.byte_tracker import BYTETracker
from src.utils.label_parser import GroundTruthObject, parse_labels
from src.visualization.visualizer import Visualizer


# Define the TrackedObject data class if not already defined
@dataclass
class TrackedObject:
    track_id: int
    label: str
    position_3d: list[float]  # [x, y, z]
    dimensions: list[float]  # [height, width, length]


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
        frame_rate=config["processing"].get("frame_rate", 30),
        feature_extractor=feature_extractor,
    )
    tracker_cars = BYTETracker(
        "config/byte_tracker_cars.yaml",
        frame_rate=config["processing"].get("frame_rate", 30),
        feature_extractor=feature_extractor,
    )
    tracker_cyclists = BYTETracker(
        "config/byte_tracker_cyclists.yaml",
        frame_rate=config["processing"].get("frame_rate", 30),
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
                    position_3d=position_3d,
                    dimensions=dimensions,
                )
                tracked_objects.append(tracked_obj)

            if has_label:
                # Retrieve ground truth objects for the current frame
                gt_objects = ground_truth_labels.get(frame_idx, [])

                # Evaluate the current frame
                evaluator.evaluate(tracked_objects, gt_objects)

            # Annotate frame with tracks using Visualizer
            annotated_frame = visualizer.display(
                img_left=img_left,
                img_right=img_right,
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


class Evaluator:
    """Evaluate the performance of the object tracking system."""

    def __init__(self):
        """Initialize the Evaluator for computing and reporting metrics."""
        self.logger = logging.getLogger("autonomous_perception.evaluation")
        if not self.logger.hasHandlers():
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        self.reset()
        self.logger.debug("Evaluator initialized and reset.")

    def reset(self) -> None:
        """Reset the evaluation metrics."""
        self.logger.debug("Resetting evaluation metrics.")
        self.precisions = []
        self.recalls = []
        self.f1_scores = []
        self.motas = []
        self.motps = []
        self.class_precisions = defaultdict(list)
        self.class_recalls = defaultdict(list)
        self.dimension_errors = []

    def evaluate(
        self,
        tracked_objects: list[TrackedObject],
        labels: list[GroundTruthObject],
    ) -> None:
        """Compare tracked objects with ground truth labels to compute metrics."""
        self.logger.debug(
            (
                f"Starting evaluation with {len(tracked_objects)} tracked objects and "
                f"{len(labels)} ground truth objects."
            ),
        )

        # Compute overall precision and recall
        precision, recall = compute_precision_recall(tracked_objects, labels)
        self.logger.debug(f"Computed Precision: {precision:.4f}, Recall: {recall:.4f}")

        # Compute F1-Score
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        self.logger.debug(f"Computed F1-Score: {f1:.4f}")

        # Compute MOTA and MOTP
        mota, motp = compute_mota_motp(tracked_objects, labels)
        self.logger.debug(f"Computed MOTA: {mota:.4f}, MOTP: {motp:.4f}")

        # Compute per-class precision and recall
        class_metrics = compute_precision_recall_per_class(tracked_objects, labels)
        self.logger.debug(f"Computed per-class precision and recall: {class_metrics}")

        # Compute dimension estimation error
        dimension_error = self._compute_dimension_error(tracked_objects, labels)
        self.logger.debug(f"Computed Dimension Error: {dimension_error:.4f} meters")

        # Append metrics to respective lists
        self.precisions.append(precision)
        self.recalls.append(recall)
        self.f1_scores.append(f1)
        self.motas.append(mota)
        self.motps.append(motp)
        self.dimension_errors.append(dimension_error)

        # Append per-class metrics
        for cls, metrics in class_metrics.items():
            self.class_precisions[cls].append(metrics["precision"])
            self.class_recalls[cls].append(metrics["recall"])
            self.logger.debug(
                f"Class '{cls}' - Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}",
            )

    def _compute_dimension_error(
        self,
        tracked_objects: list[TrackedObject],
        labels: list[GroundTruthObject],
    ) -> float:
        """Compute average dimension estimation error based on closest position_3d."""
        self.logger.debug("Starting computation of dimension estimation error.")
        total_error = 0.0
        count = 0

        used_gt = set()
        MIN_THRESHOLD = 2.0  # meters

        for tr_obj in tracked_objects:
            self.logger.debug(f"Processing TrackedObject with position_3d: {tr_obj.position_3d}")
            closest_gt = None
            min_distance = float("inf")

            for gt in labels:
                if gt.track_id in used_gt:
                    self.logger.debug(
                        f"Skipping GroundTruthObject {gt.track_id} as it has already been matched.",
                    )
                    continue
                distance = np.linalg.norm(np.array(tr_obj.position_3d) - np.array(gt.location))
                self.logger.debug(
                    f"Distance between TrackedObject and GroundTruthObject {gt.track_id}: {distance:.4f}",
                )
                if distance < min_distance:
                    min_distance = distance
                    closest_gt = gt

            if closest_gt and min_distance < MIN_THRESHOLD:
                self.logger.debug(
                    f"Matched TrackedObject to GroundTruthObject {closest_gt.track_id}"
                    f"with distance {min_distance:.4f}.",
                )
                gt_dims = closest_gt.dimensions  # [h, w, l]
                tr_dims = tr_obj.dimensions
                self.logger.debug(
                    f"GroundTruth dimensions: {gt_dims}, Tracked dimensions: {tr_dims}",
                )

                # Compute absolute difference
                error = np.abs(np.array(gt_dims) - np.array(tr_dims))
                self.logger.debug(f"Dimension error: {error}")

                # Average the error across dimensions
                mean_error = np.mean(error)
                self.logger.debug(
                    f"Mean dimension error: {mean_error:.4f} meters",
                )

                total_error += mean_error
                count += 1
                used_gt.add(closest_gt.track_id)
            else:
                self.logger.debug(
                    f"No matching GroundTruthObject found within threshold for TrackedObject"
                    f"at position {tr_obj.position_3d}.",
                )

        average_error = float(total_error / count) if count > 0 else 0.0
        self.logger.debug(
            f"Average Dimension Error: {average_error:.4f} meters over {count} matched objects.",
        )
        return average_error

    def report(self) -> None:
        """Aggregate and report overall evaluation metrics."""
        self.logger.debug("Generating evaluation report.")
        if not self.precisions:
            self.logger.warning("No evaluation data to report.")
            return

        # Compute average metrics
        avg_precision = sum(self.precisions) / len(self.precisions)
        avg_recall = sum(self.recalls) / len(self.recalls)
        avg_f1 = sum(self.f1_scores) / len(self.f1_scores)
        avg_mota = sum(self.motas) / len(self.motas)
        avg_motp = sum(self.motps) / len(self.motps)
        avg_dim_error = sum(self.dimension_errors) / len(self.dimension_errors)

        self.logger.info(f"Average Precision: {avg_precision:.4f}")
        self.logger.info(f"Average Recall: {avg_recall:.4f}")
        self.logger.info(f"Average F1-Score: {avg_f1:.4f}")
        self.logger.info(f"Average MOTA: {avg_mota:.4f}")
        self.logger.info(f"Average MOTP: {avg_motp:.4f}")
        self.logger.info(f"Average Dimension Error: {avg_dim_error:.4f} meters")

        # Report per-class metrics
        for cls in self.class_precisions:
            avg_cls_precision = sum(self.class_precisions[cls]) / len(self.class_precisions[cls])
            avg_cls_recall = sum(self.class_recalls[cls]) / len(self.class_recalls[cls])
            self.logger.info(
                f"Class '{cls}' - Average Precision: {avg_cls_precision:.4f}, Average Recall: {avg_cls_recall:.4f}",
            )


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
