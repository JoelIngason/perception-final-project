# src/main.py
import argparse

import yaml

""" from src.calibration.rectification import Rectifier """
from src.calibration.stereo_calibration import StereoCalibrator
""" from src.classification.classifier import Classifier
from src.data_loader.dataset import DataLoader
from src.detection.object_detector import ObjectDetector
from src.evaluation.evaluator import Evaluator
from src.tracking.object_tracker import ObjectTracker
from src.utils.logger import setup_logger
from src.visualization.visualizer import Visualizer """


def main(config_path):
    # Load configuration
    """ with open(config_path) as file:
        config = yaml.safe_load(file)

    # Setup logger
    logger = setup_logger('autonomous_perception', config['logging']['config_file'])
    logger.info("Starting Autonomous Perception Pipeline")

    # Initialize components
    data_loader = DataLoader(config['data']) """
    calibrator = StereoCalibrator(config['calibration'])

    """ detector = ObjectDetector(config['detection'])
    tracker = ObjectTracker(config['tracking'])
    classifier = Classifier(config['classification'])
    evaluator = Evaluator()
    visualizer = Visualizer() """

    calibration_images = config['data']['calibration_images_path']
    # Load and calibrate data
    calib_params = calibrator.calibrate(calibration_images)
    # Load and rectify data
    """ use_rectified = config['processing']['use_rectified_data']
    data = None
    rectifier = Rectifier(calib_params)
    if use_rectified:
        data = data_loader.load_sequences(calib_params)
        # Images are already rectified
    else:
        print("gg")
        #data = data_loader.load_sequences_raw(calib_params)
        # Apply rectification if necessary

    # Detection and Tracking Loop
    for frame in data:  # type: ignore
        detections = detector.detect(frame.images)
        tracked_objects = tracker.update(detections)
        classifications = classifier.classify(tracked_objects, frame.images)
        evaluator.evaluate(tracked_objects, frame.labels)
        visualizer.display(frame, tracked_objects, classifications)

    # Final Evaluation
    evaluator.report()
    logger.info("Autonomous Perception Pipeline Completed") """

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Autonomous Perception Pipeline")
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to configuration file')
    args = parser.parse_args()
    main(args.config)
