import argparse
import logging

import cv2
import yaml
from matplotlib.pylab import f

# Importing necessary classes from respective modules
from src.calibration.rectification import Rectifier
from src.calibration.stereo_calibration import StereoCalibrator
from src.data_loader.dataset import DataLoader  # Assuming the correct import path
from src.detection.object_detector import ObjectDetector
from src.evaluation.evaluator import Evaluator
from src.tracking.object_tracker import ObjectTracker
from src.utils.logger import setup_logger
from src.visualization.visualizer import Visualizer


def main(config_path: str):
    """
    Main function to execute the Autonomous Perception Pipeline.

    Args:
        config_path (str): Path to the YAML configuration file.

    """
    # Load configuration
    with open(config_path) as file:
        config = yaml.safe_load(file)
    logging.getLogger("autonomous_perception").info(f"Configuration loaded from {config_path}")

    # Setup logger
    logger = setup_logger("autonomous_perception", config["logging"]["config_file"])
    logger.info("Logger setup completed")

    # Initialize StereoCalibrator
    calibrator = StereoCalibrator(config["calibration"])
    logger.info("StereoCalibrator initialized successfully")

    # Perform calibration to obtain calibration parameters
    calibration_file = (
        config["calibration"]["calibration_file_rect"]
        if config["processing"].get("use_rectified_data", True)
        else config["calibration"]["calibration_file_raw"]
    )
    logger.info(
        f"Using {'rectified' if config['processing'].get('use_rectified_data', True) else 'raw'} calibration file: {calibration_file}",
    )
    calib_params = calibrator.calibrate(calibration_file)
    logger.info("Calibration completed successfully")

    # Initialize DataLoader
    data_loader = DataLoader(config["data"])
    logger.info("DataLoader initialized successfully")

    # Initialize Evaluator
    evaluator = Evaluator()
    logger.info("Evaluator initialized successfully")

    # Initialize Visualizer
    visualizer = Visualizer()
    logger.info("Visualizer initialized successfully")

    # Initialize Rectifier
    rectifier = Rectifier(calib_params)
    logger.info("Rectifier initialized successfully")

    # Determine whether to use rectified data
    use_rectified = config["processing"].get("use_rectified_data", True)
    logger.info(f"Configuration set to use {'rectified' if use_rectified else 'raw'} data.")

    # Load data
    if use_rectified:
        data = data_loader.load_sequences_rect(calib_params)
        logger.info("Loaded rectified data sequences.")
    else:
        data = data_loader.load_sequences_raw(calib_params)
        logger.info("Loaded raw data sequences.")
        # Apply rectification if necessary
        for seq_id in range(len(data)):
            data[str(seq_id + 1)] = rectifier.rectify_data(data[str(seq_id + 1)])

        logger.info("Applied rectification to raw data.")

    # Check if data is loaded
    if not data:
        logger.error("No data loaded. Exiting pipeline.")
        return

    # Detection and Tracking Loop
    try:
        for idx, seq in enumerate(data, 1):  # Enumerate for tracking progress
            detector = ObjectDetector(config, calib_params)
            for frame in data[seq]:
                images = frame.images  # Tuple of (left_image, right_image)
                detections, tracked_objects = detector.detect_and_track(images)

                # Evaluate tracking performance
                evaluator.evaluate(tracked_objects, frame.labels)

                # Visualize results
                visualizer.display(frame, tracked_objects)

                logger.debug(f"Processed frame {idx}/{len(data)}")

    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user. Exiting...")
    except Exception as e:
        logger.error(f"Unexpected error during processing loop: {e}")
    finally:
        # Final Evaluation
        try:
            evaluator.report()
            logger.info("Evaluator reported successfully")
        except Exception as eval_e:
            logger.error(f"Failed to generate evaluation report: {eval_e}")

        logger.info("Autonomous Perception Pipeline Completed")
        # Ensure all OpenCV windows are closed
        cv2.destroyAllWindows()


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
