import argparse
import logging
import logging.config
import pickle
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import torch
import yaml
from scipy.optimize import linear_sum_assignment

# Ensure the project modules are in the path
sys.path.append("")
from src.calibration.stereo_calibration import StereoCalibrator
from src.data_loader.dataset import DataLoader, TrackedObject
from src.depth_estimation.depth_estimator import DepthEstimator
from src.evaluation.evaluator import Evaluator
from src.models.feature_extractor.feature_extractor import FeatureExtractor
from src.models.model import load_model
from src.tracking.byte_tracker import BYTETracker
from src.utils.label_parser import GroundTruthObject, parse_labels


def is_light_color(color: tuple[int, int, int]) -> bool:
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
def run_tracking_pipeline(
    detections: dict[str, dict[int, list[TrackedObject]]],
    config_path: str,
    config_pedestrians: argparse.Namespace,
    config_cars: argparse.Namespace,
    config_cyclists: argparse.Namespace,
    ground_truth_labels: dict[int, list[GroundTruthObject]],
    evaluator: Evaluator,
    feature_extractor: FeatureExtractor,
    names: list[str],
) -> None:
    """
    Run the tracking pipeline with precomputed detections and depth maps.

    Args:
        detections (Dict[str, Dict[int, List[DatasetTrackedObject]]]): Precomputed detections per frame.
        config_path (str): Path to the configuration file.
        config_pedestrians (argparse.Namespace): BYTETracker config for pedestrians.
        config_cars (argparse.Namespace): BYTETracker config for cars.
        config_cyclists (argparse.Namespace): BYTETracker config for cyclists.
        ground_truth_labels (Dict[int, list[GroundTruthObject]]): Ground truth labels per frame.
        evaluator (Evaluator): Evaluator instance.
        feature_extractor (FeatureExtractor): Feature extractor instance.
        names (list[str]): list of class names from YOLO model.

    """
    # Load configuration
    with open(config_path) as file:
        config = yaml.safe_load(file)

    # Initialize BYTETrackers for each class
    tracker_pedestrians = BYTETracker(
        config_pedestrians,
        frame_rate=config["processing"].get("frame_rate", 10),
        feature_extractor=feature_extractor,
    )
    tracker_cars = BYTETracker(
        config_cars,
        frame_rate=config["processing"].get("frame_rate", 10),
        feature_extractor=feature_extractor,
    )
    tracker_cyclists = BYTETracker(
        config_cyclists,
        frame_rate=config["processing"].get("frame_rate", 10),
        feature_extractor=feature_extractor,
    )

    logger = logging.getLogger("BYTETrackerTuner")
    logger.info("BYTETracker initialized.")

    for seq_id, det in detections.items():
        for frame_idx, frame_detections in det.items():
            # Prepare detections for each class
            detections_pedestrians = []
            detections_cars = []
            detections_cyclists = []
            mask_pedestrians = []
            mask_cars = []
            mask_cyclists = []
            for det in frame_detections:
                bbox = det.bbox  # Assuming bbox is [x_center, y_center, width, height]
                depth, centroid = det.depth, det.centroid  # Assuming these are precomputed
                has_depth = False if depth is None or np.isnan(depth) else True

                if (
                    depth is not None and not np.isnan(depth) and 0 < depth < 6
                ):  # Example valid depth range
                    detection = [bbox[0], bbox[1], depth, bbox[2], bbox[3], det.score, det.cls]
                else:
                    detection = [bbox[0], bbox[1], 0.0, bbox[2], bbox[3], det.score, det.cls]

                if det.cls == 0:
                    detections_pedestrians.append(detection)
                    if has_depth:
                        mask_pedestrians.append([True, True, True, True, True])
                    else:
                        mask_pedestrians.append([True, True, False, True, True])
                elif det.cls == 2:
                    detections_cyclists.append(detection)
                    if has_depth:
                        mask_cyclists.append([True, True, True, True, True])
                    else:
                        mask_cyclists.append([True, True, False, True, True])
                elif det.cls == 1:
                    detections_cars.append(detection)
                    if has_depth:
                        mask_cars.append([True, True, True, True, True])
                    else:
                        mask_cars.append([True, True, False, True, True])

            # Convert to numpy arrays
            detections_pedestrians = np.array(detections_pedestrians, dtype=np.float32)
            detections_cars = np.array(detections_cars, dtype=np.float32)
            detections_cyclists = np.array(detections_cyclists, dtype=np.float32)

            # Update trackers
            active_tracks_pedestrians = tracker_pedestrians.update(
                detections_pedestrians,
                measurement_masks=mask_pedestrians,  # Use masks for depth filtering
                img=frame_detections[0].orig_img,
            )
            active_tracks_cars = tracker_cars.update(
                detections_cars,
                measurement_masks=mask_cars,  # Use masks for depth filtering
                img=frame_detections[0].orig_img,
            )
            active_tracks_cyclists = tracker_cyclists.update(
                detections_cyclists,
                measurement_masks=mask_cyclists,  # Use masks for depth filtering
                img=frame_detections[0].orig_img,
            )

            # Aggregate active tracks
            # Only add tracks if there are any otherwise we get a numpy error
            active_tracks = []
            if len(active_tracks_pedestrians) > 0:
                active_tracks.extend(active_tracks_pedestrians)
            if len(active_tracks_cars) > 0:
                active_tracks.extend(active_tracks_cars)
            if len(active_tracks_cyclists) > 0:
                active_tracks.extend(active_tracks_cyclists)
            logger.debug(f"Frame {frame_idx}: {len(active_tracks)} active tracks.")

            # Collect all active and lost tracks
            all_tracks = (
                tracker_pedestrians.tracked_stracks
                + tracker_pedestrians.lost_stracks
                + tracker_cars.tracked_stracks
                + tracker_cars.lost_stracks
                + tracker_cyclists.tracked_stracks
                + tracker_cyclists.lost_stracks
            )

            # Convert active tracks to TrackedObject instances for evaluation
            tracked_objects = []
            for track in all_tracks:
                # Extract position_3d as center coordinates
                x1, y1, z, x2, y2 = track.xyzxy
                cls = track.cls

                tracked_obj = TrackedObject(
                    track_id=int(track.track_id),
                    label=names[int(cls)],
                    bbox=[x1, y1, x2, y2],
                )
                tracked_objects.append(tracked_obj)

            # Evaluate the tracking performance

            labels = ground_truth_labels.get(frame_idx, [])
            evaluator.evaluate(tracked_objects, labels)


class BYTETrackerHyperparameterTuner:
    def __init__(
        self,
        calib_params: dict[str, Any],
        config_path: str,
        detections: dict[str, dict[int, list[TrackedObject]]],
        ground_truth_labels: dict[int, list[GroundTruthObject]],
        names: list[str],
        feature_extractor: FeatureExtractor,
        logger: logging.Logger,
        n_trials: int = 50,
        timeout: int | None = None,
    ):
        """
        Initialize the BYTETrackerHyperparameterTuner.

        Args:
            calib_params (Dict[str, Any]): Calibration parameters.
            config_path (str): Path to the configuration file.
            detections (Dict[str, Dict[int, List[TrackedObject]]]): Precomputed detections per frame.
            ground_truth_labels (Dict[int, list[GroundTruthObject]]): Ground truth labels per frame.
            names (list[str]): list of class names from YOLO model.
            feature_extractor (FeatureExtractor): Feature extractor instance.
            logger (logging.Logger): Logger instance.
            n_trials (int, optional): Number of optimization trials.
            timeout (int, optional): Time limit for optimization in seconds.

        """
        self.calib_params = calib_params
        self.config_path = config_path
        self.detections = detections
        self.ground_truth_labels = ground_truth_labels
        self.names = names
        self.feature_extractor = feature_extractor
        self.logger = logger
        self.n_trials = n_trials
        self.timeout = timeout

    def objective(self, trial: optuna.Trial) -> float:
        """
        Objective function for Optuna.

        Args:
            trial (optuna.Trial): Optuna trial object.

        Returns:
            float: Negative MOTA score (since Optuna minimizes by default).

        """
        # try:
        # Suggest hyperparameters for each class with unique names
        config_pedestrians = {
            "alpha": trial.suggest_float("ped_alpha", 0.01, 2.0),
            "beta": trial.suggest_float("ped_beta", 0.01, 2.0),
            "alpha_second": trial.suggest_float("ped_alpha_second", 0.01, 2.0),
            "beta_second": trial.suggest_float("ped_beta_second", 0.01, 2.0),
            "max_avg_velocity_x": trial.suggest_float("ped_max_avg_velocity_x", 0.1, 400.0),
            "max_avg_velocity_y": trial.suggest_float("ped_max_avg_velocity_y", 0.1, 400.0),
            "track_buffer": trial.suggest_int("ped_track_buffer", 1, 15),
            "match_thresh": trial.suggest_float("ped_match_thresh", 0.4, 0.99),
            "match_second_thresh": trial.suggest_float("ped_match_second_thresh", 0.4, 0.99),
            "match_final_thresh": trial.suggest_float("ped_match_final_thresh", 0.4, 0.99),
            "track_high_thresh": trial.suggest_float("ped_track_high_thresh", 0.1, 0.99),
            "track_low_thresh": trial.suggest_float("ped_track_low_thresh", 0.1, 0.7),
            "new_track_thresh": trial.suggest_float("ped_new_track_thresh", 0.1, 0.9),
            "fuse_score": trial.suggest_categorical("ped_fuse_score", [True, False]),
            "remove_stationary": trial.suggest_categorical("ped_remove_stationary", [True, False]),
        }

        config_cars = {
            "alpha": trial.suggest_float("car_alpha", 0.01, 2.0),
            "beta": trial.suggest_float("car_beta", 0.01, 2.0),
            "alpha_second": trial.suggest_float("car_alpha_second", 0.01, 2.0),
            "beta_second": trial.suggest_float("car_beta_second", 0.01, 2.0),
            "max_avg_velocity_x": trial.suggest_float("car_max_avg_velocity_x", 0.1, 400.0),
            "max_avg_velocity_y": trial.suggest_float("car_max_avg_velocity_y", 0.1, 400.0),
            "track_buffer": trial.suggest_int("car_track_buffer", 1, 15),
            "match_thresh": trial.suggest_float("car_match_thresh", 0.4, 0.99),
            "match_second_thresh": trial.suggest_float("car_match_second_thresh", 0.4, 0.99),
            "match_final_thresh": trial.suggest_float("car_match_final_thresh", 0.4, 0.99),
            "track_high_thresh": trial.suggest_float("car_track_high_thresh", 0.1, 0.99),
            "track_low_thresh": trial.suggest_float("car_track_low_thresh", 0.1, 0.7),
            "new_track_thresh": trial.suggest_float("car_new_track_thresh", 0.1, 0.9),
            "fuse_score": trial.suggest_categorical("car_fuse_score", [True, False]),
            "remove_stationary": trial.suggest_categorical("car_remove_stationary", [True, False]),
        }

        config_cyclists = {
            "alpha": trial.suggest_float("cyclist_alpha", 0.01, 2.0),
            "beta": trial.suggest_float("cyclist_beta", 0.01, 2.0),
            "alpha_second": trial.suggest_float("cyclist_alpha_second", 0.01, 2.0),
            "beta_second": trial.suggest_float("cyclist_beta_second", 0.01, 2.0),
            "max_avg_velocity_x": trial.suggest_float("cyclist_max_avg_velocity_x", 0.1, 400.0),
            "max_avg_velocity_y": trial.suggest_float("cyclist_max_avg_velocity_y", 0.1, 400.0),
            "track_buffer": trial.suggest_int("cyclist_track_buffer", 1, 15),
            "match_thresh": trial.suggest_float("cyclist_match_thresh", 0.2, 0.99),
            "match_second_thresh": trial.suggest_float("cyclist_match_second_thresh", 0.2, 0.99),
            "match_final_thresh": trial.suggest_float("cyclist_match_final_thresh", 0.2, 0.99),
            "track_high_thresh": trial.suggest_float("cyclist_track_high_thresh", 0.1, 0.99),
            "track_low_thresh": trial.suggest_float("cyclist_track_low_thresh", 0.1, 0.7),
            "new_track_thresh": trial.suggest_float("cyclist_new_track_thresh", 0.1, 0.9),
            "fuse_score": trial.suggest_categorical("cyclist_fuse_score", [True, False]),
            "remove_stationary": trial.suggest_categorical(
                "cyclist_remove_stationary",
                [True, False],
            ),
        }

        self.logger.info(f"Trial {trial.number}: Suggested hyperparameters.")

        # Initialize Evaluator for this trial
        evaluator = Evaluator(distance_threshold=1000.0)

        # Run the tracking pipeline with current hyperparameters
        run_tracking_pipeline(
            detections=self.detections,
            config_path=self.config_path,
            config_pedestrians=argparse.Namespace(**config_pedestrians),
            config_cars=argparse.Namespace(**config_cars),
            config_cyclists=argparse.Namespace(**config_cyclists),
            ground_truth_labels=self.ground_truth_labels,
            evaluator=evaluator,
            feature_extractor=self.feature_extractor,
            names=self.names,
        )

        # Get the average MOTA
        avg_mota = evaluator.get_average_mota()
        avg_precision = evaluator.get_average_precision()
        avg_extra_track_rate = evaluator.get_average_extra_tracks_rate()
        avg_dimension_error = evaluator.get_average_dimension_error()

        self.logger.info(
            f"Trial {trial.number}: Average MOTA = {avg_mota:.4f} and Precision = {avg_precision:.4f}",
        )

        # Since Optuna minimizes by default, return negative MOTA and Precision,
        # with a penalty for extra tracks and dimension error
        return (
            -1.0 * (avg_mota + avg_precision) + 2.0 * avg_extra_track_rate
            # + 0.1 * avg_dimension_error
        )

    # except Exception as e:
    #    self.logger.error(f"Trial {trial.number} failed with exception: {e}")
    #    return float("inf")  # Assign a large error to failed trials

    def tune(self) -> optuna.Study:
        """
        Run the hyperparameter tuning.

        Returns:
            optuna.Study: The Optuna study object after optimization.

        """
        study = optuna.create_study(
            direction="minimize",
            study_name="BYTETracker_Hyperparameter_Tuning",
            sampler=optuna.samplers.TPESampler(seed=42),
        )
        study.optimize(
            self.objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            n_jobs=8,
            show_progress_bar=True,
        )
        self.logger.info(
            f"Study completed. Best trial: {study.best_trial.number} with MOTA {-study.best_trial.value:.4f}",
        )
        self.logger.info("Best hyperparameters:")
        for key, value in study.best_trial.params.items():
            self.logger.info(f"  {key}: {value}")
        return study


def precompute_detections_and_depth(
    yolo_model: Any,
    depth_estimator: DepthEstimator,
    data_loader: DataLoader,
    use_rectified_data: bool,
    calib_params: dict[str, Any],
) -> dict[str, dict[int, list[TrackedObject]]]:
    """
    Precompute detections for all frames.

    Args:
        yolo_model (Any): Loaded YOLO model.
        depth_estimator (DepthEstimator): DepthEstimator instance.
        data_loader (DataLoader): DataLoader instance.
        use_rectified_data (bool): Whether to use rectified data.
        calib_params (Dict[str, Any]): Calibration parameters.

    Returns:
        Dict[str, Dict[int, List[TrackedObject]]]: Precomputed detections per frame.

    """
    detections = {}
    depth_maps = {}
    sequences = (
        data_loader.load_sequences_rect(calib_params)
        if use_rectified_data
        else data_loader.load_sequences_raw(calib_params)
    )
    logger = logging.getLogger("Precompute")
    logger.info(f"Precomputing detections and depth maps for {len(sequences)} sequences.")

    # first check if pickled detections and depth maps are available

    detections_file = "cache/detections.pkl"
    depth_maps_file = "cache/depth_maps.pkl"
    if Path(detections_file).is_file() and Path(depth_maps_file).is_file():
        logger.info("Loading precomputed detections and depth maps.")
        with open(detections_file, "rb") as f:
            detections = pickle.load(f)
        with open(depth_maps_file, "rb") as f:
            depth_maps = pickle.load(f)
        return detections

    for seq_id, frames in sequences.items():
        # Skip sequence 3
        if seq_id == "3":
            continue
        detections_seq = {}
        depth_maps_seq = {}
        for frame_idx, frame in enumerate(frames, 1):
            img_left = frame.image_left
            img_right = frame.image_right

            # Compute disparity and depth maps
            disparity_map = depth_estimator.compute_disparity_map(img_left, img_right)
            depth_map = depth_estimator.compute_depth_map(disparity_map, img_left)

            # Perform object detection
            results = yolo_model(img_left)[0].boxes.cpu().numpy()

            frame_detections = []
            for result in results:
                bbox = result.xyxy[0].tolist()  # [x_center, y_center, width, height]
                # Assuming depth_estimator can compute depth and centroid from bbox and depth_map
                depth, centroid = depth_estimator.compute_object_depth(bbox, depth_map)
                tracked_obj = TrackedObject(
                    frame_idx=frame_idx,
                    bbox=bbox,
                    depth=depth,
                    centroid=centroid,
                    score=result.conf.item() if hasattr(result, "conf") else 1.0,
                    cls=int(result.cls.item()) if hasattr(result, "cls") else 0,
                    sequence=seq_id,
                    orig_img=img_left,
                )
                frame_detections.append(tracked_obj)

            depth_maps_seq[frame_idx] = depth_map
            detections_seq[frame_idx] = frame_detections

            if frame_idx % 100 == 0:
                logger.info(f"Processed {frame_idx} frames in sequence {seq_id}.")
        depth_maps["seq_id"] = depth_maps_seq
        detections["seq_id"] = detections_seq

    logger.info("Precomputing completed.")
    # Save the precomputed detections and depth maps
    with open(detections_file, "wb") as f:
        pickle.dump(detections, f)
    with open(depth_maps_file, "wb") as f:
        pickle.dump(depth_maps, f)
    return detections


def main():
    # Configure logging
    logging.basicConfig(
        level=logging.WARNING,  # Set to INFO or DEBUG as needed
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger("BYTETrackerTuner")

    # Argument parser for command line execution
    parser = argparse.ArgumentParser(description="BYTETracker Hyperparameter Tuner")
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to the main configuration YAML file",
    )
    parser.add_argument(
        "--n_trials",
        type=int,
        default=3000,
        help="Number of hyperparameter optimization trials",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Timeout for hyperparameter optimization in seconds",
    )
    args = parser.parse_args()

    # Load configuration
    config_path = args.config
    if not Path(config_path).is_file():
        logger.error(f"Configuration file not found at {config_path}")
        sys.exit(1)
    with open(config_path) as file:
        config = yaml.safe_load(file)

    # Initialize StereoCalibrator and perform calibration
    calibrator = StereoCalibrator(config["calibration"])
    calibration_file_key = (
        "calibration_file_rect"
        if config["processing"].get("use_rectified_data", True)
        else "calibration_file_raw"
    )
    calibration_file = config["calibration"].get(calibration_file_key)
    if not calibration_file or not Path(calibration_file).is_file():
        logger.error(f"Calibration file not found at {calibration_file}")
        sys.exit(1)
    calib_params = calibrator.calibrate(calibration_file)

    # Initialize DataLoader
    data_loader = DataLoader(config["data"])

    # Initialize YOLO model with tracking
    yolo = load_model(config["models"]["classification"]["path"])
    if not hasattr(yolo, "names"):
        logger.error("Loaded YOLO model does not have a 'names' attribute.")
        sys.exit(1)
    names = yolo.names  # Ensure the model has a 'names' attribute
    logger.info(f"YOLO model loaded with classes: {names}")

    # Initialize DepthEstimator
    depth_estimator = DepthEstimator(calib_params, config)

    # Precompute detections and depth maps
    detections = precompute_detections_and_depth(
        yolo_model=yolo,
        depth_estimator=depth_estimator,
        data_loader=data_loader,
        use_rectified_data=config["processing"].get("use_rectified_data", True),
        calib_params=calib_params,
    )

    # Load ground truth labels
    label_files = config["data"].get("label_files", [])
    if not label_files:
        logger.error(
            "Label file paths not specified in the configuration under 'data.label_files'.",
        )
        sys.exit(1)

    ground_truth_labels = {}
    for label_file in label_files:
        if label_file == "":
            continue
        parsed_labels = parse_labels(label_file)
        ground_truth_labels.update(parsed_labels)

    # Initialize FeatureExtractor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    feature_extractor = FeatureExtractor(device=device)
    logger.info(f"FeatureExtractor initialized on device: {device}")

    # Initialize HyperparameterTuner
    tuner = BYTETrackerHyperparameterTuner(
        calib_params=calib_params,
        config_path=config_path,
        detections=detections,
        ground_truth_labels=ground_truth_labels,
        names=["Pedestrian", "Car", "Cyclist"],
        feature_extractor=feature_extractor,
        logger=logger,
        n_trials=args.n_trials,
        timeout=args.timeout,
    )

    # Run hyperparameter tuning
    study = tuner.tune()

    # Retrieve the best hyperparameters
    best_params = study.best_trial.params
    logger.info("Best hyperparameters found:")
    for key, value in best_params.items():
        logger.info(f"  {key}: {value}")

    # Optionally, save the best hyperparameters to a separate file
    best_params_path = "config/best_byte_tracker_params.yaml"
    with open(best_params_path, "w") as f:
        yaml.dump(best_params, f)
    logger.info(f"Best hyperparameters saved to {best_params_path}")


if __name__ == "__main__":
    main()
