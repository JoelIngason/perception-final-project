import argparse
import logging
import logging.config

import cv2
import yaml

from src.calibration.stereo_calibration import StereoCalibrator
from src.data_loader.dataset import DataLoader
from src.depth_estimation.depth_estimator import DepthEstimator
from src.detection.detector import ObjectDetector
from src.evaluation.evaluator import Evaluator
from src.tracking.tracker import ObjectTracker
from src.utils.label_parser import GroundTruthObject, parse_labels
from src.visualization.visualizer import Visualizer


def setup_logger(name: str, config_file: str | None = None) -> logging.Logger:
    """Set up logging based on the provided configuration."""
    if config_file:
        with open(config_file) as f:
            config = yaml.safe_load(f)
            logging.config.dictConfig(config)
    logger = logging.getLogger(name)
    return logger


def main(config_path: str) -> None:
    """Execute the Enhanced Autonomous Perception Pipeline."""
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

    if not calibration_file:
        logger.error(
            f"Calibration file path '{calibration_file_key}' not specified in configuration.",
        )
        return

    calib_params = calibrator.calibrate(calibration_file)

    # Initialize components
    data_loader = DataLoader(config["data"])
    evaluator = Evaluator()
    visualizer = Visualizer()
    depth_estimator = DepthEstimator(calib_params, config)

    # Load data
    data = (
        data_loader.load_sequences_rect(calib_params)
        if use_rectified_data
        else data_loader.load_sequences_raw(calib_params)
    )
    if not data:
        logger.error("No data loaded. Exiting pipeline.")
        return

    # Initialize detector and trackers
    detector = ObjectDetector(config, calib_params)
    tracker_left = ObjectTracker(config, calib_params, "left")
    tracker_right = ObjectTracker(config, calib_params, "right")

    # Parse ground truth labels
    label_file = config["data"].get(
        "label_file",
        "data/34759_final_project_rect/seq_02/labels.txt",
    )  # Update path accordingly
    ground_truth_dict = parse_labels(label_file)
    logger.info(f"Parsed ground truth labels from {label_file}")

    # Make OpenCV window resizable
    cv2.namedWindow("Autonomous Perception", cv2.WINDOW_NORMAL)

    try:
        for seq_id, frames in data.items():
            logger.info(f"Processing sequence {seq_id} with {len(frames)} frames.")
            evaluator.reset()
            visualizer.reset()
            tracker_left.reset()
            tracker_right.reset()

            for frame_idx, frame in enumerate(frames, 1):
                try:
                    logger.debug(
                        f"Processing frame {frame_idx}/{len(frames)} in sequence {seq_id}.",
                    )

                    # Convert images to BGR if they are in RGB
                    img_left = frame.image_left
                    img_right = frame.image_right
                    if img_left.shape[2] == 3:
                        img_left = cv2.cvtColor(img_left, cv2.COLOR_RGB2BGR)
                    if img_right.shape[2] == 3:
                        img_right = cv2.cvtColor(img_right, cv2.COLOR_RGB2BGR)

                    # Perform detection on left and right images
                    detections_left = detector.detect(img_left, "left")
                    detections_right = detector.detect(img_right, "right")
                    logger.debug(
                        f"Detected {len(detections_left)} objects in the left image and "
                        f"{len(detections_right)} objects in the right image.",
                    )

                    # Compute disparity and depth maps
                    disparity_map = depth_estimator.compute_disparity_map(img_left, img_right)
                    depth_map = depth_estimator.compute_depth_map(disparity_map)

                    # Add 3d information to detections
                    for detection in detections_left + detections_right:
                        median_depth, object_depth = depth_estimator.compute_object_depth(
                            detection,
                            depth_map,
                        )
                        detection.compute_position_3d(median_depth)
                        detection.estimate_dimensions(object_depth)

                    # Update trackers with detections
                    tracked_objects_left = tracker_left.update_tracks(detections_left)
                    tracked_objects_right = tracker_right.update_tracks(detections_right)
                    logger.debug(
                        f"Tracking {len(tracked_objects_left)} objects on left image, "
                        f"{len(tracked_objects_right)} objects on right image.",
                    )

                    # Merge tracked objects from both cameras
                    tracked_objects = tracked_objects_left + tracked_objects_right

                    # Retrieve ground truth for the current frame
                    # Adjust frame numbering if necessary (assuming labels start at 0)
                    gt_objects = ground_truth_dict.get(frame_idx - 1, [])

                    # Convert ground truth labels to GroundTruthObject instances if not already
                    gt_objects_parsed = [
                        GroundTruthObject(
                            frame=gt.frame,
                            track_id=gt.track_id,
                            obj_type=gt.obj_type,
                            truncated=gt.truncated,
                            occluded=gt.occluded,
                            alpha=gt.alpha,
                            bbox=gt.bbox,
                            dimensions=gt.dimensions,
                            location=gt.location,
                            rotation_y=gt.rotation_y,
                            score=gt.score,
                        )
                        for gt in gt_objects
                    ]

                    # Evaluate tracking
                    evaluator.evaluate(tracked_objects, gt_objects_parsed)

                    # Log per-frame evaluation
                    logger.debug(
                        f"Frame {frame_idx}: Tracked {len(tracked_objects)} objects, "
                        f"Ground Truth {len(gt_objects_parsed)} objects.",
                    )

                    # Visualize results
                    if not visualizer.display(
                        frame,
                        tracked_objects_left,
                        tracked_objects_right,
                        depth_map,
                    ):
                        logger.info("Visualization interrupted by user. Exiting...")
                        return
                except Exception:
                    logger.exception(
                        f"Error processing frame {frame_idx} in sequence {seq_id}",
                    )
                    continue
            # After processing all frames in the sequence
            logger.info(f"Sequence {seq_id} processed successfully with {len(frames)} frames.")

            # Compute and report overall metrics
            evaluator.report()

    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user. Exiting...")
    except Exception:
        logger.exception("Unexpected error during processing")
    finally:
        cv2.destroyAllWindows()
        logger.info("Enhanced Autonomous Perception Pipeline Completed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enhanced Autonomous Perception Pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to the configuration YAML file",
    )
    args = parser.parse_args()
    main(args.config)
