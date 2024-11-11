# src/main.py
import argparse
import yaml
import logging
from src.utils.logger import setup_logger
from src.data_loader.dataset import DataLoader
from src.calibration.stereo_calibration import StereoCalibrator
from src.calibration.rectification import Rectifier
from src.detection.object_detector import ObjectDetector
from src.tracking.object_tracker import ObjectTracker
from src.classification.classifier import Classifier
from src.evaluation.evaluator import Evaluator
from src.visualization.visualizer import Visualizer

def main(config_path):
    # Load configuration
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    # Setup logger
    logger = setup_logger('autonomous_perception', config['logging']['config_file'])
    logger.info("Starting Autonomous Perception Pipeline")

    # Initialize components
    data_loader = DataLoader(config['data'])
    calibrator = StereoCalibrator(config['calibration'])
    calibration_params = calibrator.calibrate(config['data']['calibration_file'])
    rectifier = Rectifier(calibration_params)
    detector = ObjectDetector(config['detection'])
    tracker = ObjectTracker(config['tracking'])
    classifier = Classifier(config['classification'])
    evaluator = Evaluator()
    visualizer = Visualizer()

    # Load and rectify data
    use_rectified = config['processing']['use_rectified_data']
    data = None

    if use_rectified:
        data = data_loader.load_sequences(calibration_params)
        # Images are already rectified
    else:
        data = data_loader.load_sequences_raw(calibration_params)
        # Apply rectification if necessary


    # Detection and Tracking Loop
    for frame in data:
        # Rectify images if not already rectified (depending on data)
        # Assuming data is already rectified in '34759_final_project_rect'
        # If processing raw data, uncomment the following lines:
        # rectified_left, rectified_right = rectifier.rectify_images(frame.images[0], frame.images[1])
        # frame.images = (rectified_left, rectified_right)

        detections = detector.detect(frame.images)
        tracked_objects = tracker.update(detections)
        classifications = classifier.classify(tracked_objects)
        evaluator.evaluate(tracked_objects, frame.labels)
        visualizer.display(frame, tracked_objects, classifications)

    # Final Evaluation
    evaluator.report()
    logger.info("Autonomous Perception Pipeline Completed")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Autonomous Perception Pipeline")
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to configuration file')
    args = parser.parse_args()
    main(args.config)