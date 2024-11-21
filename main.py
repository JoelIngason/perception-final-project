import argparse

import cv2
import yaml

from src.calibration.rectification import Rectifier
from src.calibration.stereo_calibration import StereoCalibrator
from src.data_loader.dataset import DataLoader
from src.depth_estimation.depth_estimator import DepthEstimator
from src.detection.object_detector import ObjectDetector
from src.evaluation.evaluator import Evaluator
from src.tracking.tracker import ObjectTracker
from src.utils.logger import setup_logger
from src.visualization.visualizer import Visualizer


def main(config_path: str) -> None:
    """Execute the Autonomous Perception Pipeline."""
    with open(config_path) as file:
        config = yaml.safe_load(file)

    logger = setup_logger("autonomous_perception", config["logging"]["config_file"])
    logger.info(f"Configuration loaded from {config_path}")

    use_rectified_data = config["processing"].get("use_rectified_data", True)
    if not use_rectified_data:
        calibrator = StereoCalibrator(config["calibration"])
        calibration_file = config["calibration"].get("calibration_file_raw")
        if not calibration_file:
            logger.error("Calibration file path not specified in configuration.")
            return
        calib_params = calibrator.calibrate(calibration_file)
    else:
        calibration_file = config["calibration"].get("calibration_file_rect")
        if not calibration_file:
            logger.error("Calibration file path for rectified data not specified.")
            return
        calibrator = StereoCalibrator(config["calibration"])
        calib_params = calibrator.calibrate(calibration_file)

    data_loader = DataLoader(config["data"])
    evaluator = Evaluator()
    visualizer = Visualizer()
    tracker = ObjectTracker(config)
    depth_estimator = DepthEstimator(calib_params, config)
    rectifier = None
    if not use_rectified_data:
        rectifier = Rectifier(calib_params)
    data = (
        data_loader.load_sequences_rect(calib_params)
        if use_rectified_data
        else data_loader.load_sequences_raw(calib_params)
    )
    if rectifier:
        data = rectifier.rectify_data(data)
    if not data:
        logger.error("No data loaded. Exiting pipeline.")
        return

    detector = ObjectDetector(config, calib_params)

    try:
        for seq_id, frames in data.items():
            logger.info(f"Processing sequence {seq_id} with {len(frames)} frames.")
            detector.reset()
            evaluator.reset()
            visualizer.reset()
            for frame in frames:
                try:
                    detections, tracked_objects, depth_map = detector.detect_and_track(frame.images)
                    evaluator.evaluate(tracked_objects, frame.labels)
                    if not visualizer.display(frame, tracked_objects, depth_map):
                        return
                except Exception as e:
                    logger.exception(f"Error processing frame: {e}")
                    continue
            evaluator.report()
            logger.info(f"Sequence {seq_id} processed successfully with {len(frames)} frames.")
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user. Exiting...")
    except Exception as e:
        logger.exception(f"Unexpected error during processing: {e}")
    finally:
        evaluator.report()
        cv2.destroyAllWindows()
        logger.info("Autonomous Perception Pipeline Completed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Autonomous Perception Pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to the configuration YAML file",
    )
    args = parser.parse_args()
    main(args.config)
