import logging
import sys
from calendar import c
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import optuna
from sympy import im

sys.path.append("")
from src.utils.label_parser import GroundTruthObject


def preprocess_image(image: np.ndarray) -> np.ndarray:
    if len(image.shape) != 2:
        raise ValueError("Input image for preprocessing must be grayscale.")
    return cv2.equalizeHist(image)


class DepthEstimator:
    """DepthEstimator class to compute disparity maps and estimate depth from stereo images."""

    def __init__(
        self,
        calib_params: dict[str, np.ndarray],
        config: dict[str, Any],
        image_size: tuple[int, int] = (1280, 720),
        baseline_correction_factor: float = 1.0,
    ):
        """
        Initialize the DepthEstimator with calibration parameters and configuration.

        Args:
            calib_params (Dict[str, np.ndarray]): Stereo calibration parameters.
            config (Dict[str, Any]): Configuration parameters.
            image_size (Tuple[int, int], optional): Image size for disparity map computation.
            baseline_correction_factor (float, optional): Factor to adjust the baseline.

        """
        self.logger = logging.getLogger("DepthEstimator")
        self.calib_params = calib_params
        self.config = config
        self.image_size = image_size

        # Compute focal length and baseline
        try:
            if "P_rect_02" not in self.calib_params:
                raise KeyError("P_rect_02 not found in calibration parameters.")
            self.focal_length = self.calib_params["P_rect_02"][0, 0]

            if "T_02" not in self.calib_params or "T_03" not in self.calib_params:
                raise KeyError("Translation vectors T_02 and/or T_03 not found.")

            self.baseline = (
                np.linalg.norm(self.calib_params["T_02"] - self.calib_params["T_03"])
                * baseline_correction_factor
            )
            self.logger.info(f"Baseline (B): {self.baseline:.4f} meters")
            self.logger.info(f"Focal Length (f): {self.focal_length:.4f} pixels")
        except KeyError as e:
            self.logger.exception(f"Missing calibration parameter: {e}")
            raise
        except Exception as e:
            self.logger.exception(f"Error initializing DepthEstimator: {e}")
            raise

        # StereoSGBM parameters
        stereo_config = self.config.get("stereo", {})
        min_disparity = stereo_config.get("min_disparity", 0)
        num_disparities = stereo_config.get("num_disparities", 16 * 20)  # Must be divisible by 16
        block_size = stereo_config.get("block_size", 5)  # Typically odd numbers

        # Validate num_disparities
        if num_disparities % 16 != 0:
            self.logger.warning(
                f"num_disparities {num_disparities} is not divisible by 16. Adjusting to nearest higher multiple.",
            )
            num_disparities = ((num_disparities // 16) + 1) * 16
            self.config["stereo"]["num_disparities"] = num_disparities

        # Fine-tuned matching parameters
        P1 = stereo_config.get("P1", 8 * 3 * block_size**2)
        P2 = stereo_config.get("P2", 32 * 3 * block_size**2)
        disp12_max_diff = stereo_config.get("disp12_max_diff", 1)
        uniqueness_ratio = stereo_config.get("uniqueness_ratio", 15)
        speckle_window_size = stereo_config.get("speckle_window_size", 100)
        speckle_range = stereo_config.get("speckle_range", 32)
        pre_filter_cap = stereo_config.get("pre_filter_cap", 63)

        # Initialize StereoSGBM matcher
        try:
            self.stereo_matcher = cv2.StereoSGBM.create(
                minDisparity=min_disparity,
                numDisparities=num_disparities,
                blockSize=block_size,
                P1=P1,
                P2=P2,
                disp12MaxDiff=disp12_max_diff,
                uniquenessRatio=uniqueness_ratio,
                speckleWindowSize=speckle_window_size,
                speckleRange=speckle_range,
                preFilterCap=pre_filter_cap,
            )
            self.logger.info("StereoSGBM matcher initialized with parameters.")
        except Exception as e:
            self.logger.exception(f"Failed to initialize StereoSGBM matcher: {e}")
            raise

        ## Initialize MiDaS for depth refinement
        # try:
        #    self.midas = torch.hub.load("intel-isl/MiDaS", "DPT_Large", pretrained=True)
        #    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        #    self.transform = midas_transforms.dpt_transform
        #    self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        #    self.midas.to(self.device)
        #    self.midas.eval()
        #    self.logger.info("MiDaS model loaded successfully.")
        # except Exception as e:
        #    self.logger.exception(f"Failed to load MiDaS model: {e}")
        #    raise

    def compute_disparity_map(self, img_left: np.ndarray, img_right: np.ndarray) -> np.ndarray:
        """
        Compute the disparity map from rectified stereo images.

        Args:
            img_left (np.ndarray): Rectified left image.
            img_right (np.ndarray): Rectified right image.

        Returns:
            np.ndarray: Refined disparity map.

        """
        try:
            # Convert images to grayscale
            gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
            gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

            # Preprocessing
            gray_left = preprocess_image(gray_left)
            gray_right = preprocess_image(gray_right)

            # Compute disparity map
            disparity_map = (
                self.stereo_matcher.compute(gray_left, gray_right).astype(np.float32) / 16.0
            )

            # Post-process disparity map
            disparity_map = self.postprocess_disparity(disparity_map)

            # Check if disparity map has valid values
            if np.all(disparity_map <= 0):
                self.logger.error("Disparity map contains only invalid values.")
                raise ValueError("Invalid disparity map computed.")

            self.logger.debug("Disparity map computed successfully.")
            self.logger.debug(
                f"Disparity Map Stats: min={np.nanmin(disparity_map)}, max={np.nanmax(disparity_map)}, mean={np.nanmean(disparity_map)}",
            )

            return disparity_map

        except Exception as e:
            self.logger.exception(f"Error computing disparity map: {e}")
            raise

    def postprocess_disparity(self, disparity: np.ndarray) -> np.ndarray:
        """
        Refine disparity map using median blur and normalization.

        Args:
            disparity (np.ndarray): Raw disparity map.

        Returns:
            np.ndarray: Refined disparity map.

        """
        try:
            # Replace invalid disparities with zero
            disparity[disparity <= 0.0] = 0.0

            # Normalize disparity for median filtering
            disparity_normalized = cv2.normalize(
                disparity,
                None,
                0,
                255,
                cv2.NORM_MINMAX,
                dtype=cv2.CV_8U,
            )

            # Apply median blur
            kernel_size = 9
            refined_disparity = cv2.medianBlur(disparity_normalized, kernel_size)

            # Convert back to float32 with original scale
            refined_disparity = (
                refined_disparity.astype(np.float32) / 255.0 * (disparity.max() - disparity.min())
                + disparity.min()
            )

            self.logger.debug("Disparity post-processing applied successfully.")

            return refined_disparity

        except Exception as e:
            self.logger.exception(f"Error during disparity post-processing: {e}")
            raise

    def compute_depth_map(self, disparity_map: np.ndarray, left_image: np.ndarray) -> np.ndarray:
        """
        Compute depth map from disparity map.

        Args:
            disparity_map (np.ndarray): Disparity map.
            left_image (np.ndarray): Rectified left image.

        Returns:
            np.ndarray: Depth map in meters.

        """
        try:
            # Compute initial depth
            with np.errstate(divide="ignore", invalid="ignore"):
                depth_map = (self.focal_length * self.baseline) / (disparity_map + 1e-6)
                depth_map[disparity_map <= 0.0] = np.nan  # Mask invalid disparities
                depth_map[depth_map > 80.0] = np.nan  # Cap depth values at 80m

            self.logger.debug("Initial depth map computed successfully.")

            # Refine depth map using MiDaS
            # Optional: You can uncomment the following lines if you wish to use MiDaS for refinement
            # pred_depth_metric = self.refine_with_midas(left_image, target_size=depth_map.shape)

            # Combine stereo and MiDaS depth maps
            # combined_depth = self.combine_depth_maps(depth_map, pred_depth_metric)

            # Final refinement
            refined_depth = self.refine_depth_map(depth_map)

            return refined_depth

        except Exception as e:
            self.logger.exception(f"Error computing depth map: {e}")
            raise

    def compute_object_depth(
        self,
        bbox: np.ndarray,
        depth_map: np.ndarray,
    ) -> tuple[float, list[int]]:
        """
        Compute the median depth within the object's bounding box using histogram analysis.

        Args:
            bbox (np.ndarray): Bounding box coordinates in the format [x1, y1, x2, y2].
            depth_map (np.ndarray): Depth map in meters.

        Returns:
            Tuple[float, list[float]]: Estimated object depth and centroid.

        """
        try:
            x, y, w, h = bbox
            x, y, w, h = int(x), int(y), int(w), int(h)
            # Use a five-pixel margin around the center of the bounding box
            margin = 5

            # Extract depth values within the bounding box
            depth_values = depth_map[y + margin : y + h - margin, x + margin : x + w - margin]
            depth_values = depth_values[~np.isnan(depth_values)]

            if len(depth_values) == 0:
                self.logger.warning("No valid depth values found in the object mask.")
                return (np.nan, [x, y])

            # Use histogram analysis to find the most frequent depth
            hist, bin_edges = np.histogram(depth_values, bins=30)
            peak_index = np.argmax(hist)
            peak_value = (bin_edges[peak_index] + bin_edges[peak_index + 1]) / 2

            # if peak_value > 50.0:
            #    self.logger.warning(f"Large depth value ({peak_value:.2f}m) detected for object.")

            return (peak_value, [x, y])
        except Exception as e:
            self.logger.exception(f"Error computing object depth: {e}")
            return (np.nan, [x, y])

    def refine_depth_map(self, depth_map: np.ndarray) -> np.ndarray:
        """
        Refine the depth map using post-processing techniques to enhance quality.

        Techniques include:
            - Bilateral Filtering
            - Hole Filling

        Args:
            depth_map (np.ndarray): Raw depth map in meters.

        Returns:
            np.ndarray: Refined depth map.

        """
        try:
            refined_depth = depth_map.copy()

            # 1. Bilateral Filtering
            nan_mask = np.isnan(refined_depth)
            refined_depth[nan_mask] = 0.0
            refined_depth = cv2.bilateralFilter(
                refined_depth.astype(np.float32),
                d=9,
                sigmaColor=75,
                sigmaSpace=75,
            )
            refined_depth[nan_mask] = np.nan
            self.logger.debug("Bilateral filtering applied to depth map.")

            # 2. Hole Filling using Inpainting
            invalid_mask = np.isnan(refined_depth).astype(np.uint8) * 255
            if np.any(invalid_mask):
                inpaint_input = np.where(np.isnan(refined_depth), 0, refined_depth).astype(
                    np.float32,
                )
                inpainted_depth = cv2.inpaint(
                    inpaint_input,
                    invalid_mask,
                    inpaintRadius=3,
                    flags=cv2.INPAINT_TELEA,
                )
                inpainted_depth[invalid_mask == 255] = np.nan
                refined_depth = inpainted_depth
                self.logger.debug("Hole filling (inpainting) applied to depth map.")
            else:
                self.logger.debug("No invalid depth values found. Skipping inpainting.")

            return refined_depth

        except Exception as e:
            self.logger.exception(f"Error during depth map refinement: {e}")
            raise

    def evaluate_depth_estimation(
        self,
        images_left: list[np.ndarray],
        images_right: list[np.ndarray],
        labels: dict[int, list[GroundTruthObject]],
        show: bool = False,
    ) -> float:
        """
        Evaluate the depth estimation against ground truth labels.

        Args:
            images_left (List[np.ndarray]): List of left stereo images.
            images_right (List[np.ndarray]): List of right stereo images.
            labels (Dict): Ground truth labels for object depth.
            show (bool): Whether to display images and plots.

        Returns:
            float: Overall mean depth error.

        """
        errors = []
        disparities_vs_depths = []

        for idx, (img_left, img_right) in enumerate(zip(images_left, images_right, strict=False)):
            if img_left is None or img_right is None:
                self.logger.error(f"Could not read images at index {idx}")
                continue

            # Compute disparity map
            disparity_map = self.compute_disparity_map(img_left, img_right)

            # Compute depth map
            depth_map = self.compute_depth_map(disparity_map, img_left)

            # Refine depth map
            depth_map = self.refine_depth_map(depth_map)

            frame_errors = []
            current_labels = labels[idx] if idx in labels else []
            for gt_obj in current_labels:
                # Get ROI
                x1, y1, x2, y2 = gt_obj.bbox
                bbox = np.array([int(x1), int(y1), int(x2), int(y2)])

                # Compute estimated depth in the ROI
                estimated_depth, _ = self.compute_object_depth(bbox, depth_map)

                # Ground truth depth
                gt_z = gt_obj.location[2]

                # Compare estimated depth to ground truth
                if np.isnan(estimated_depth):
                    self.logger.warning(f"No valid depth for object at frame {idx}")
                    continue

                error = np.abs(estimated_depth - gt_z)
                frame_errors.append(error)
                disparities_vs_depths.append([1 / estimated_depth, gt_z])

                if show:
                    # Optionally display images and bounding boxes
                    print(f"bbox: {bbox}")
                    cv2.rectangle(img_left, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(
                        img_left,
                        f"Est Z: {estimated_depth:.2f}m",
                        (int(x1), int(y1 - 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        1,
                    )
                    cv2.putText(
                        img_left,
                        f"GT Z: {gt_z:.2f}m",
                        (int(x1), int(y1 - 40)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 255),
                        1,
                    )

            if frame_errors:
                mean_error = np.mean(frame_errors)
                errors.append(mean_error)
            else:
                errors.append(np.nan)

            if show:
                cv2.imshow("Image with Depth", img_left)
                cv2.waitKey(1)

        # Compute overall mean error
        overall_mean_error = np.nanmean(errors)
        self.logger.info(f"Overall mean depth error: {overall_mean_error:.2f}m")

        # Optionally plot errors over frames
        if show:
            plt.figure()
            plt.plot(errors)
            plt.xlabel("Frame")
            plt.ylabel("Mean Depth Error (m)")
            plt.title("Depth Estimation Error Over Frames")
            plt.show()

            # Plot disparity vs depth
            disparities_vs_depths = np.array(disparities_vs_depths)
            plt.figure()
            plt.scatter(disparities_vs_depths[:, 0], disparities_vs_depths[:, 1], s=3)
            plt.xlabel("Inverse Estimated Depth (1/m)")
            plt.ylabel("Ground Truth Depth (m)")
            plt.title("Inverse Estimated Depth vs Ground Truth Depth")
            plt.show()

        return overall_mean_error


import numpy as np

# Assuming DepthEstimator and other necessary classes are already imported


class HyperparameterTuner:
    def __init__(
        self,
        calib_params: dict[str, np.ndarray],
        config: dict[str, Any],
        images_left: list[np.ndarray],
        images_right: list[np.ndarray],
        labels: dict[int, list[GroundTruthObject]],
        logger: logging.Logger,
        n_trials: int = 50,
        timeout: int = None,
    ):
        """
        Initialize the HyperparameterTuner.

        Args:
            calib_params (dict[str, np.ndarray]): Calibration parameters.
            config (dict[str, Any]): Configuration dictionary.
            images_left (List[np.ndarray]): Left stereo images.
            images_right (List[np.ndarray]): Right stereo images.
            labels (Dict[int, List[GroundTruthObject]]): Ground truth labels.
            logger (logging.Logger): Logger instance.
            n_trials (int, optional): Number of optimization trials.
            timeout (int, optional): Time limit for optimization in seconds.

        """
        self.calib_params = calib_params
        self.config = config
        self.images_left = images_left
        self.images_right = images_right
        self.labels = labels
        self.logger = logger
        self.n_trials = n_trials
        self.timeout = timeout

    def objective(self, trial: optuna.Trial) -> float:
        """
        Objective function for Optuna.

        Args:
            trial (optuna.Trial): Optuna trial object.

        Returns:
            float: Mean depth estimation error.

        """
        # Suggest hyperparameters
        stereo_config = {}
        stereo_config["min_disparity"] = trial.suggest_int("min_disparity", 0, 30)
        stereo_config["num_disparities"] = trial.suggest_int("num_disparities", 16, 512, step=16)
        stereo_config["block_size"] = trial.suggest_int("block_size", 3, 21, step=2)  # must be odd

        # P1 and P2 depend on block_size
        window_size = stereo_config["block_size"]
        stereo_config["P1"] = trial.suggest_int(
            "P1",
            1 * 3 * window_size**2,
            42 * 3 * window_size**2,
        )
        stereo_config["P2"] = trial.suggest_int(
            "P2",
            2 * 3 * window_size**2,
            192 * 3 * window_size**2,
        )

        stereo_config["disp12_max_diff"] = trial.suggest_int("disp12_max_diff", 0, 20)
        stereo_config["uniqueness_ratio"] = trial.suggest_int("uniqueness_ratio", 1, 20)
        stereo_config["speckle_window_size"] = trial.suggest_int("speckle_window_size", 5, 200)
        stereo_config["speckle_range"] = trial.suggest_int("speckle_range", 1, 64)
        stereo_config["pre_filter_cap"] = trial.suggest_int("pre_filter_cap", 5, 63)

        # Update config with suggested hyperparameters
        config_copy = self.config.copy()
        config_copy["stereo"] = stereo_config

        # Initialize DepthEstimator with current hyperparameters
        depth_estimator = DepthEstimator(
            calib_params=self.calib_params,
            config=config_copy,
            image_size=self.config.get("image_size", (1224, 370)),
            baseline_correction_factor=self.config.get("baseline_correction_factor", 1.0),
        )

        # Evaluate depth estimation without showing plots
        # Suppress logging during trials to reduce verbosity
        original_level = self.logger.level
        self.logger.setLevel(logging.WARNING)
        try:
            # Retrieve the overall mean error
            overall_mean_error = depth_estimator.evaluate_depth_estimation(
                self.images_left,
                self.images_right,
                self.labels,
                show=False,
            )
        except Exception as e:
            self.logger.error(f"Trial failed with exception: {e}")
            return float("inf")  # Assign a large error to failed trials
        finally:
            self.logger.setLevel(original_level)

        return overall_mean_error

    def tune(self) -> optuna.Study:
        """
        Run the hyperparameter tuning.

        Returns:
            optuna.Study: The Optuna study object after optimization.

        """
        study = optuna.create_study(
            direction="minimize",
            study_name="DepthEstimator_Hyperparameter_Tuning",
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
            f"Study completed. Best trial: {study.best_trial.number} with error {study.best_trial.value:.4f}",
        )
        self.logger.info("Best hyperparameters:")
        for key, value in study.best_trial.params.items():
            self.logger.info(f"  {key}: {value}")
        return study


if __name__ == "__main__":
    import logging
    import sys
    from pathlib import Path

    import yaml

    from src.calibration.stereo_calibration import StereoCalibrator
    from src.data_loader.dataset import DataLoader
    from src.utils.label_parser import parse_labels

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,  # Set to INFO to reduce verbosity during tuning
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger("Main")

    # Paths
    config_path = "config/config.yaml"
    calibration_file = "data/34759_final_project_rect/calib_cam_to_cam.txt"
    labels_file = "data/34759_final_project_rect/seq_01/labels.txt"

    # Load configuration
    if not Path(config_path).is_file():
        raise FileNotFoundError(f"Configuration file not found at {config_path}")
    with open(config_path) as file:
        config = yaml.safe_load(file)

    # Initialize StereoCalibrator and perform calibration
    calibrator = StereoCalibrator(config)
    calib_params = calibrator.calibrate(calibration_file)

    # Load images
    data_loader = DataLoader(config)
    images = data_loader.load_sequences_rect(calib_params=calib_params)
    images_left = [frame.image_left for frame in images["1"]]
    images_right = [frame.image_right for frame in images["1"]]

    # Load labels
    labels = parse_labels(labels_file)

    # Initialize DepthEstimator with initial config
    depth_estimator = DepthEstimator(calib_params, config)

    # Optional: Split data into training and validation sets
    # tuning_labels = {idx: labels[idx] for idx in range(0, len(images_left), 10)}
    ##images_left = images_left[:100]
    # images_right = images_right[:100]

    tuning_labels = labels
    # Initialize HyperparameterTuner
    tuner = HyperparameterTuner(
        calib_params=calib_params,
        config=config,
        images_left=images_left,
        images_right=images_right,
        labels=tuning_labels,
        logger=logger,
        n_trials=1000,  # Adjust the number of trials as needed
    )

    # Run hyperparameter tuning
    study = tuner.tune()

    # Retrieve the best hyperparameters
    best_params = study.best_trial.params
    logger.info("Best hyperparameters found:")
    for key, value in best_params.items():
        logger.info(f"  {key}: {value}")

    # Update the config with the best hyperparameters
    config["stereo"] = best_params

    # Save the updated configuration for future use
    tuned_config_path = "config/tuned_config.yaml"
    with open(tuned_config_path, "w") as file:
        yaml.dump(config, file)
    logger.info(f"Tuned configuration saved to {tuned_config_path}")

    # Initialize DepthEstimator with tuned hyperparameters
    tuned_depth_estimator = DepthEstimator(calib_params, config)

    # Perform final evaluation with tuned parameters
    tuned_depth_estimator.evaluate_depth_estimation(
        images_left,
        images_right,
        labels,
        show=True,  # Set to True to visualize results
    )
