import logging

import cv2
import numpy as np

from src.detection.detection_result import DetectionResult


class DepthEstimator:
    """DepthEstimator class to compute disparity maps and estimate depth from stereo images."""

    def __init__(self, calib_params: dict[str, np.ndarray], config: dict) -> None:
        """
        Initialize the DepthEstimator with calibration parameters and configuration.

        Args:
            calib_params (Dict[str, np.ndarray]): Stereo calibration parameters.
            config (Dict): Configuration parameters.

        """
        self.logger = logging.getLogger("autonomous_perception.depth_estimation")
        self.calib_params = calib_params
        self.config = config

        # Compute focal length and baseline
        try:
            # Focal length-
            self.focal_length = self.calib_params["P_rect_02"][0, 0]

            # Baseline is provided directly
            # self.baseline = 0.54  # meters
            self.baseline = self.calib_params["baseline"]
            self.logger.info(f"Baseline (B): {self.baseline} meters")

        except KeyError as e:
            self.logger.exception(f"Missing calibration parameter: {e}")
            raise
        except Exception as e:
            self.logger.exception(f"Error initializing DepthEstimator: {e}")
            raise

        # StereoSGBM parameters
        min_disparity = self.config["stereo"].get("min_disparity", 0)
        num_disparities = self.config["stereo"].get(
            "num_disparities",
            160,  # Must be divisible by 16
        )
        block_size = self.config["stereo"].get("block_size", 5)  # Reduced for finer details

        # Fine-tuned matching parameters
        P1 = 8 * 3 * block_size**2
        P2 = 32 * 3 * block_size**2
        disp12_max_diff = self.config["stereo"].get("disp12_max_diff", 1)
        uniqueness_ratio = self.config["stereo"].get("uniqueness_ratio", 15)
        speckle_window_size = self.config["stereo"].get("speckle_window_size", 100)
        speckle_range = self.config["stereo"].get("speckle_range", 32)
        pre_filter_cap = self.config["stereo"].get("pre_filter_cap", 63)
        mode = cv2.STEREO_SGBM_MODE_SGBM_3WAY

        # Initialize StereoSGBM matcher
        # TODO check for cuda and then use cv2.cuda.
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
            mode=mode,
        )
        self.logger.info("StereoSGBM matcher initialized with parameters.")

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
            gray_left = (
                cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY) if len(img_left.shape) == 3 else img_left
            )
            gray_right = (
                cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)
                if len(img_right.shape) == 3
                else img_right
            )

            # Compute disparity map
            disparity_map = (
                self.stereo_matcher.compute(gray_left, gray_right).astype(np.float32) / 16.0
            )

            # Post-process disparity map
            disparity_map = cv2.medianBlur(disparity_map, 5)
            disparity_map[disparity_map <= 0.0] = np.nan  # Mask invalid disparities

            return disparity_map

        except Exception as e:
            self.logger.exception(f"Error computing disparity map: {e}")
            raise

    def compute_depth_map(self, disparity_map: np.ndarray) -> np.ndarray:
        """
        Compute depth map from disparity map.

        Args:
            disparity_map (np.ndarray): Disparity map.

        Returns:
            np.ndarray: Depth map in meters.

        """
        try:
            # Compute depth using the formula: Depth = (focal_length * baseline) / disparity
            depth_map = (self.focal_length * self.baseline) / disparity_map
            depth_map[np.isnan(depth_map) | np.isinf(depth_map) | (depth_map <= 0.0)] = np.nan

            return depth_map

        except Exception as e:
            self.logger.exception(f"Error computing depth map: {e}")
            raise

    def compute_object_depth(
        self,
        detection: DetectionResult,
        depth_map: np.ndarray,
    ) -> tuple[float, float]:
        """
        Compute the median depth within the object's mask, excluding outliers.

        also compute the depth of the object in meters.
        """
        if detection.mask is None:
            self.logger.debug("No mask available for depth computation.")
            return (-1.0, 0.0)

        # Ensure the mask is a NumPy array with valid coordinates
        mask_coords = np.array(detection.mask)
        if mask_coords.ndim != 2 or mask_coords.shape[1] != 2:
            self.logger.error("detection.mask should be a 2D array with shape (N, 2).")
            return (-1.0, 0.0)

        # Extract x and y coordinates from the mask
        x_coords = mask_coords[:, 0] - 1
        y_coords = mask_coords[:, 1] - 1

        # Extract depth values corresponding to the mask
        depth_values = depth_map[y_coords, x_coords]

        # Filter out invalid depth values (non-finite or non-positive)
        valid_depths = depth_values[np.isfinite(depth_values) & (depth_values > 0)]
        if valid_depths.size == 0:
            self.logger.warning(f"No valid depth values for detection {detection.label}")
            return (-1.0, 0.0)

        # Remove outliers using the interquartile range (IQR) method
        q1 = np.percentile(valid_depths, 25)
        q3 = np.percentile(valid_depths, 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        filtered_depths = valid_depths[
            (valid_depths >= lower_bound) & (valid_depths <= upper_bound)
        ]

        if filtered_depths.size == 0:
            self.logger.warning(
                f"All depth values are considered outliers for detection {detection.label}",
            )
            return (-1.0, 0.0)

        # Calculate the median of the filtered depth values
        median_depth = float(np.median(filtered_depths))
        self.logger.debug(
            f"Median Depth for {detection.label} after outlier removal: {median_depth:.2f}m",
        )

        # Calculate the depth of the object in meters
        object_depth = np.max(filtered_depths) - np.min(filtered_depths)

        return (median_depth, object_depth)
