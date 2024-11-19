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
            P_rect_02 = calib_params["P_rect_02"]  # 3x4 matrix
            P_rect_03 = calib_params["P_rect_03"]  # 3x4 matrix
            self.focal_length = P_rect_02[0, 0]  # fx from rectified left projection matrix

            # Compute baseline
            Tx_left = P_rect_02[0, 3] / -self.focal_length  # In meters
            Tx_right = P_rect_03[0, 3] / -self.focal_length  # In meters
            self.baseline = Tx_right - Tx_left  # Positive value

            self.logger.info(f"Focal length set to {self.focal_length}")
            self.logger.info(f"Baseline set to {self.baseline} meters")
        except KeyError as e:
            self.logger.exception(f"Missing calibration parameter: {e}")
            raise
        except Exception as e:
            self.logger.exception(f"Error initializing DepthEstimator: {e}")
            raise

        # Initialize StereoSGBM matcher with improved parameters
        self.stereo_matcher = cv2.StereoSGBM.create(
            minDisparity=0,
            numDisparities=16 * 5,  # Must be divisible by 16
            blockSize=5,
            P1=8 * 3 * 5**2,
            P2=32 * 3 * 5**2,
            disp12MaxDiff=1,
            uniquenessRatio=15,
            speckleWindowSize=0,
            speckleRange=2,
            preFilterCap=63,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
        )
        self.logger.info("StereoSGBM matcher initialized with improved parameters")

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
            # Convert images to grayscale
            gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
            gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

            # Compute disparity using StereoSGBM
            disparity = self.stereo_matcher.compute(gray_left, gray_right).astype(np.float32) / 16.0
            self.logger.debug("Disparity map computed.")

            # Replace negative disparities with zero
            disparity[disparity < 0] = 0

            return disparity
        except Exception as e:
            self.logger.exception(f"Error computing disparity map: {e}")
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
        depth = self.focal_length * self.baseline / disparity
        return depth
