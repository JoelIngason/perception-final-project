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

        # StereoSGBM parameters
        window_size = 5
        min_disparity = 0
        num_disparities = 16 * 10  # Must be divisible by 16
        block_size = window_size
        P1 = 8 * 3 * window_size**2
        P2 = 32 * 3 * window_size**2
        disp12_max_diff = 1
        uniqueness_ratio = 10
        speckle_window_size = 100
        speckle_range = 32
        pre_filter_cap = 63
        mode = cv2.STEREO_SGBM_MODE_SGBM_3WAY

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
        self.logger.info("StereoSGBM matcher initialized with improved parameters")

        # Initialize WLS filter for disparity map refinement
        self.wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=self.stereo_matcher)
        self.right_matcher = cv2.ximgproc.createRightMatcher(self.stereo_matcher)
        self.wls_filter.setLambda(8000.0)
        self.wls_filter.setSigmaColor(1.5)
        self.logger.info("WLS filter initialized for disparity refinement")

    def compute_disparity_map(self, img_left: np.ndarray, img_right: np.ndarray) -> np.ndarray:
        """
        Compute the refined disparity map from rectified stereo images.

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

            # Compute left and right disparity maps
            disparity_left = self.stereo_matcher.compute(gray_left, gray_right)
            disparity_right = self.right_matcher.compute(gray_right, gray_left)

            # Apply WLS filter to refine disparity map
            disparity_left = np.int16(disparity_left)
            disparity_right = np.int16(disparity_right)
            filtered_disparity = self.wls_filter.filter(
                disparity_left,
                img_left,
                None,
                disparity_right,
            )
            self.logger.debug("Disparity map computed and refined using WLS filter.")

            ## Normalize the disparity map for visualization
            # filtered_disparity = cv2.normalize(
            #    filtered_disparity,
            #    None,
            #    alpha=0,
            #    beta=255,
            #    norm_type=cv2.NORM_MINMAX,
            #    dtype=cv2.CV_8U,
            # )

        except Exception as e:
            self.logger.exception(f"Error computing disparity map: {e}")
            raise
        else:
            return filtered_disparity

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
