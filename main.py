import argparse

import yaml

from src.calibration.rectification import Rectifier
from src.calibration.stereo_calibration import StereoCalibrator
from src.data_loader.dataset import DataLoader
from src.detection.object_detector import ObjectDetector
from src.evaluation.evaluator import Evaluator
from src.utils.logger import setup_logger
from src.visualization.visualizer import Visualizer


def main(config_path: str):
    # Load configuration
    with open(config_path) as file:
        config = yaml.safe_load(file)

    # Setup logger
    logger = setup_logger("autonomous_perception", config["logging"]["config_file"])
    logger.info("Starting Autonomous Perception Pipeline")

    # Initialize components
    data_loader = DataLoader(config["data"])
    calibrator = StereoCalibrator(config["calibration"])

    detector = ObjectDetector(config["detection"])
    # Removed ObjectTracker as tracking is integrated within ObjectDetector
    evaluator = Evaluator()
    visualizer = Visualizer()

    use_rectified = config["processing"].get("use_rectified_data", True)

    calibration_file = (
        config["calibration"]["calibration_file_rect"]
        if use_rectified
        else config["calibration"]["calibration_file_raw"]
    )
    logger.info(f"Using {'rectified' if use_rectified else 'raw'} data.")

    # Load and calibrate data
    calib_params = calibrator.calibrate(calibration_file)

    # Load and rectify data
    rectifier = Rectifier(calib_params)
    if use_rectified:
        data = data_loader.load_sequences(calib_params)
        logger.info("Loaded rectified data sequences.")
    else:
        data = data_loader.load_sequences_raw(calib_params)
        logger.info("Loaded raw data sequences.")
        # Apply rectification if necessary
        # Assuming load_sequences_raw returns raw images that need rectification
        data = rectifier.rectify_data(data)
        logger.info("Applied rectification to raw data.")

    if data is None:
        logger.error("No data loaded. Exiting pipeline.")
        return

    # Detection and Tracking Loop
    for frame in data:  # Assuming 'data' is an iterable of frames
        try:
            images = frame.images  # Tuple of (left_image, right_image)
            detections, tracked_objects = detector.detect_and_track(images)

            # Evaluate tracking performance
            evaluator.evaluate(tracked_objects, frame.labels)

            # Visualize results
            visualizer.display(frame, tracked_objects)

        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            continue  # Continue with the next frame

    # Final Evaluation
    evaluator.report()
    logger.info("Autonomous Perception Pipeline Completed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Autonomous Perception Pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to configuration file",
    )
    args = parser.parse_args()

    main(args.config)
