import argparse

import cv2
import yaml

from src.calibration.rectification import Rectifier
from src.calibration.stereo_calibration import StereoCalibrator
from src.data_loader.dataset import DataLoader
from src.detection.object_detector import ObjectDetector
from src.evaluation.evaluator import Evaluator
from src.utils.logger import setup_logger
from src.visualization.visualizer import Visualizer


def main(config_path: str) -> None:  # noqa: C901, PLR0915
    """
    Execute the Autonomous Perception Pipeline.

    Args:
        config_path (str): Path to the YAML configuration file.

    """
    # Load configuration
    with open(config_path) as file:
        config = yaml.safe_load(file)

    # Setup logger
    logger = setup_logger("autonomous_perception", config["logging"]["config_file"])
    logger.info(f"Configuration loaded from {config_path}")

    # Initialize StereoCalibrator
    calibrator = StereoCalibrator(config["calibration"])
    calibration_file = (
        config["calibration"].get("calibration_file_rect")
        if config["processing"].get("use_rectified_data", True)
        else config["calibration"].get("calibration_file_raw")
    )
    if not calibration_file:
        logger.error("Calibration file path not specified in configuration.")
        return

    calib_params = calibrator.calibrate(calibration_file)
    logger.info("Calibration completed successfully")

    # Initialize DataLoader
    data_loader = DataLoader(config["data"])
    logger.info("DataLoader initialized successfully")

    # Initialize other components
    evaluator = Evaluator()
    visualizer = Visualizer()
    rectifier = (
        Rectifier(calib_params)
        if not config["processing"].get("use_rectified_data", True)
        else None
    )

    # Load data
    if config["processing"].get("use_rectified_data", True):
        data = data_loader.load_sequences_rect(calib_params)
        logger.info("Loaded rectified data sequences.")
    else:
        data = data_loader.load_sequences_raw(calib_params)
        logger.info("Loaded raw data sequences.")
        if rectifier:
            data = rectifier.rectify_data(data)
            logger.info("Applied rectification to raw data.")

    if not any(data.values()):
        logger.error("No data loaded. Exiting pipeline.")
        return

    # Initialize ObjectDetector
    detector = ObjectDetector(config, calib_params)

    # Detection and Tracking Loop
    try:
        for seq_id, frames in data.items():
            logger.info(f"Processing sequence {seq_id} with {len(frames)} frames.")
            detector.reset()
            evaluator.reset()
            visualizer.reset()
            for frame in frames:
                try:
                    # Perform detection and tracking
                    detections, tracked_objects, disparity_map = detector.detect_and_track(
                        frame.images,
                    )

                    # Evaluate
                    evaluator.evaluate(tracked_objects, frame.labels)

                    # Visualize
                    visualizer.display(frame, tracked_objects, disparity_map)
                except Exception:
                    logger.exception("Error processing frame")
                    continue
            evaluator.report()
            logger.info(f"Sequence {seq_id} processed successfully with {len(frames)} frames.")

    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user. Exiting...")
    except Exception:
        logger.exception("Unexpected error during processing")
    finally:
        # Final Evaluation
        try:
            evaluator.report()
            logger.info("Evaluator reported successfully")
        except Exception:
            logger.exception("Failed to generate evaluation report")
        # Ensure all OpenCV windows are closed
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
