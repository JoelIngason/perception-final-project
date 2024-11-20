import logging

import cv2
import numpy as np


class DepthEstimator:
    """DepthEstimator class to compute disparity maps and estimate depth from stereo images."""

    def __init__(self, calib_params: dict[str, np.ndarray], logger: logging.Logger) -> None:
        """
        Initialize the DepthEstimator with calibration parameters.

        Args:
            calib_params (Dict[str, np.ndarray]): Stereo calibration parameters.
            logger (logging.Logger): Logger for logging messages.
        """
        self.logger = logger
        self.calib_params = calib_params

        # Compute focal length and baseline from rectified projection matrices
        try:
            P_rect_02 = calib_params["P_rect_02"]  # Left camera projection matrix
            P_rect_03 = calib_params["P_rect_03"]  # Right camera projection matrix

            # Extract focal length from P_rect_02
            self.focal_length = P_rect_02[0, 0]  # fx from the left camera
            self.logger.info(f"Focal Length (fx): {self.focal_length}")

            # Compute baseline (in meters)
            # Note: P[0,3] = -fx * baseline
            self.baseline = -(P_rect_03[0, 3] - P_rect_02[0, 3]) / self.focal_length
            self.logger.info(f"Baseline (B): {self.baseline} meters")
        except KeyError as e:
            self.logger.exception(f"Missing calibration parameter: {e}")
            raise
        except Exception as e:
            self.logger.exception(f"Error initializing DepthEstimator: {e}")
            raise

        # StereoSGBM parameters (optimized for better depth estimation)
        window_size = 3  # Increased for better matching
        min_disparity = -16  # Allow negative disparities
        num_disparities = 16 * 8  # Increased range for better depth resolution
        block_size = window_size

        # Adjusted P1 and P2 for better smoothness
        P1 = 8 * 3 * window_size**2
        P2 = 64 * 3 * window_size**2  # Increased for stronger smoothness

        # Fine-tuned matching parameters
        disp12_max_diff = 2  # Increased tolerance
        uniqueness_ratio = 10  # More strict uniqueness check
        speckle_window_size = 100  # Reduced for faster processing
        speckle_range = 16  # Reduced for less noise
        pre_filter_cap = 31  # Reduced to prevent over-saturation
        mode = cv2.STEREO_SGBM_MODE_HH  # Higher quality mode

        # Initialize StereoSGBM matcher with improved parameters
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
            np.ndarray: Disparity map.
        """
        try:
            # Convert images to grayscale if they are not already
            if len(img_left.shape) == 3:
                gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
            else:
                gray_left = img_left

            if len(img_right.shape) == 3:
                gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)
            else:
                gray_right = img_right

            # Compute disparity map
            disparity_map = self.stereo_matcher.compute(gray_left, gray_right)

            # Convert disparity map to floating-point values
            disparity_map = disparity_map.astype(np.float32) / 16.0

            # Handle invalid disparities by setting them to the average disparity of valid pixels around them
            invalid_mask = disparity_map <= 0
            disparity_map = cv2.inpaint(
                disparity_map,
                invalid_mask.astype(np.uint8),
                inpaintRadius=3,
                flags=cv2.INPAINT_TELEA,
            )

            # Log disparity map statistics
            self.logger.info(
                f"Disparity Map Stats - Min: {np.nanmin(disparity_map)}, Max: {np.nanmax(disparity_map)}, Mean: {np.nanmean(disparity_map)}"
            )

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
            # Compute depth map
            depth_map = (self.focal_length * self.baseline) / disparity_map

            # Handle invalid depth values
            depth_map[np.isnan(depth_map) | np.isinf(depth_map)] = 0.0

            # Log depth map statistics
            self.logger.info(
                f"Depth Map Stats - Min: {np.min(depth_map)}, Max: {np.max(depth_map)}, Mean: {np.mean(depth_map)}"
            )

            return depth_map

        except Exception as e:
            self.logger.exception(f"Error computing depth map: {e}")
            raise

    def compute_depth(self, disparity: float) -> float:
        """
        Compute depth from disparity value.

        Args:
            disparity (float): Disparity value.

        Returns:
            float: Estimated depth in meters.
        """
        if disparity <= 0:
            self.logger.warning(f"Non-positive disparity encountered: {disparity}")
            return 0.0
        depth = (self.focal_length * self.baseline) / disparity
        return depth
