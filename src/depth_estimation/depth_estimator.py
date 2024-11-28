import logging
from typing import Any

import cv2
import numpy as np
from scipy.stats import zscore

from src.detection.detection_result import DetectionResult


class DepthEstimator:
    """DepthEstimator class to compute disparity maps and estimate depth from stereo images."""

    def __init__(self, calib_params: dict[str, np.ndarray], config: dict[str, Any]) -> None:
        """
        Initialize the DepthEstimator with calibration parameters and configuration.

        Args:
            calib_params (Dict[str, np.ndarray]): Stereo calibration parameters.
            config (Dict[str, Any]): Configuration parameters.

        """
        self.logger = logging.getLogger("autonomous_perception.depth_estimation")
        self.calib_params = calib_params
        self.config = config

        # Compute focal length and baseline
        try:
            # Focal length from the projection matrix P_rect_02
            if "P_rect_02" not in self.calib_params:
                raise KeyError("P_rect_02 not found in calibration parameters.")
            self.focal_length = self.calib_params["P_rect_02"][0, 0]

            # Compute baseline from translation vectors T_02 and T_03
            if "T_02" not in self.calib_params or "T_03" not in self.calib_params:
                raise KeyError(
                    "Translation vectors T_02 and/or T_03 not found in calibration parameters.",
                )

            # Assuming T_02 and T_03 are the translation vectors for left and right cameras
            # Baseline is the distance between the two cameras
            self.baseline = np.linalg.norm(self.calib_params["T_02"] - self.calib_params["T_03"])
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
        P1 = 8 * 3 * block_size**2
        P2 = 32 * 3 * block_size**2
        disp12_max_diff = stereo_config.get("disp12_max_diff", 1)
        uniqueness_ratio = stereo_config.get("uniqueness_ratio", 15)
        speckle_window_size = stereo_config.get("speckle_window_size", 100)
        speckle_range = stereo_config.get("speckle_range", 32)
        pre_filter_cap = stereo_config.get("pre_filter_cap", 63)
        mode = cv2.STEREO_SGBM_MODE_SGBM_3WAY

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
                mode=mode,
            )
            self.logger.info("StereoSGBM matcher initialized with parameters.")
        except Exception as e:
            self.logger.exception(f"Failed to initialize StereoSGBM matcher: {e}")
            raise

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
            # Convert images to grayscale if they are not already
            gray_left = (
                img_left if len(img_left.shape) == 2 else cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
            )
            gray_right = (
                img_right
                if len(img_right.shape) == 2
                else cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)
            )

            # Compute disparity map
            disparity_map = (
                self.stereo_matcher.compute(gray_left, gray_right).astype(np.float32) / 16.0
            )

            # Post-process disparity map
            disparity_map = cv2.medianBlur(disparity_map, 5)
            disparity_map[disparity_map <= 0.0] = np.nan  # Mask invalid disparities

            self.logger.debug("Disparity map computed successfully.")
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
            with np.errstate(divide="ignore", invalid="ignore"):
                depth_map = (self.focal_length * self.baseline) / disparity_map
                depth_map[disparity_map <= 0.0] = np.nan  # Mask invalid disparities

            self.logger.debug("Depth map computed successfully.")
            depth_map = self.refine_depth_map(depth_map)
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
        Compute the median depth within the object's mask, excluding extreme outliers.

        Uses z-score filtering to exclude depth values that deviate significantly from the mean.

        Args:
            detection (DetectionResult): Detection result containing the object's mask.
            depth_map (np.ndarray): Depth map in meters.

        Returns:
            Tuple[float, float]: Median depth and object depth range in meters.

        """
        if not detection.mask.any():
            self.logger.debug("No mask available for depth computation.")
            return (-1.0, 0.0)

        # Convert mask to NumPy array for efficient processing
        mask_coords = np.array(detection.mask)
        if mask_coords.ndim != 2 or mask_coords.shape[1] != 2:
            self.logger.error("detection.mask should be a 2D array with shape (N, 2).")
            return (-1.0, 0.0)

        # Extract x and y coordinates from the mask (1-based indexing)
        x_coords = mask_coords[:, 0] - 1  # Convert to 0-based indexing
        y_coords = mask_coords[:, 1] - 1  # Convert to 0-based indexing

        # Validate coordinates
        height, width = depth_map.shape
        valid_indices = (x_coords >= 0) & (x_coords < width) & (y_coords >= 0) & (y_coords < height)
        if not np.any(valid_indices):
            self.logger.warning("All mask coordinates are out of bounds after indexing adjustment.")
            return (-1.0, 0.0)

        # Apply valid indices and convert to integer type for indexing
        x_coords = x_coords[valid_indices].astype(int)
        y_coords = y_coords[valid_indices].astype(int)

        # Extract depth values corresponding to the mask
        try:
            depth_values = depth_map[y_coords, x_coords]
        except IndexError as e:
            self.logger.exception(f"Error accessing depth map with mask coordinates: {e}")
            return (-1.0, 0.0)

        # Filter out invalid depth values (non-finite or non-positive)
        valid_depths = depth_values[np.isfinite(depth_values) & (depth_values > 0)]
        if valid_depths.size == 0:
            self.logger.warning(f"No valid depth values for detection '{detection.label}'.")
            return (-1.0, 0.0)

        # Compute z-scores for depth values
        z_scores = zscore(valid_depths)
        if np.isnan(z_scores).all():
            self.logger.warning(f"Z-score computation failed for detection '{detection.label}'.")
            return (-1.0, 0.0)

        # Exclude outliers with a z-score threshold (e.g., |z| > 2.5)
        z_threshold = self.config.get("stereo", {}).get("z_threshold", 2.5)
        filtered_depths = valid_depths[np.abs(z_scores) <= z_threshold]

        if filtered_depths.size == 0:
            self.logger.warning(
                f"All depth values are excluded as outliers for detection '{detection.label}'.",
            )
            return (-1.0, 0.0)

        # Calculate the median of the filtered depth values
        median_depth = float(np.median(filtered_depths))
        self.logger.debug(
            f"Median Depth for '{detection.label}' after excluding outliers: {median_depth:.2f}m",
        )

        # Calculate the depth range of the object (difference between max and min filtered depth)
        object_depth = float(np.max(filtered_depths) - np.min(filtered_depths))
        self.logger.debug(f"Depth Range for '{detection.label}': {object_depth:.2f}m")

        if object_depth > 10.0:
            self.logger.warning(
                f"Large depth range ({object_depth:.2f}m) detected for object '{detection.label}'.",
            )
        return (median_depth, object_depth)

    def refine_depth_map(self, depth_map: np.ndarray) -> np.ndarray:
        """
        Refine the depth map using post-processing techniques to enhance quality.

        Techniques include:
            - Bilateral Filtering
            - Guided Filtering
            - Hole Filling

        Args:
            depth_map (np.ndarray): Raw depth map in meters.

        Returns:
            np.ndarray: Refined depth map.

        """
        try:
            refined_depth = depth_map.copy()

            # 1. Bilateral Filtering

            # Apply bilateral filter
            refined_depth = cv2.bilateralFilter(
                refined_depth,
                d=9,
                sigmaColor=75,
                sigmaSpace=75,
            )
            refined_depth[depth_map == 0] = np.nan  # Restore NaNs

            self.logger.debug("Bilateral filtering applied to depth map.")

            # 2. Hole Filling using Inpainting
            # Create mask of invalid depth values
            invalid_mask = np.isnan(refined_depth).astype(np.uint8) * 255
            if np.any(invalid_mask):
                # Inpaint using Telea's method
                inpainted_depth = cv2.inpaint(
                    refined_depth,
                    invalid_mask,
                    inpaintRadius=3,
                    flags=cv2.INPAINT_TELEA,
                )
                refined_depth = inpainted_depth
                refined_depth[refined_depth <= 0.0] = np.nan  # Mask invalid disparities again
                self.logger.debug("Hole filling (inpainting) applied to depth map.")
            else:
                self.logger.debug("No invalid depth values found. Skipping inpainting.")

            return refined_depth

        except Exception as e:
            self.logger.exception(f"Error during depth map refinement: {e}")
            raise

    def visualize_depth_map(
        self,
        img: np.ndarray,
        depth_map: np.ndarray,
        window_name: str = "Depth Map",
        save_path: str | None = None,
    ) -> None:
        """
        Visualize the depth map alongside the original image.

        Args:
            img (np.ndarray): Original rectified left image.
            depth_map (np.ndarray): Refined depth map in meters.
            window_name (str, optional): Name of the display window.
            save_path (str, optional): Path to save the visualized depth map image.

        """
        try:
            # Normalize depth map for visualization
            depth_min = np.nanmin(depth_map)
            depth_max = np.nanmax(depth_map)
            normalized_depth = np.copy(depth_map)
            normalized_depth[np.isnan(normalized_depth)] = (
                depth_max  # Replace NaNs for visualization
            )
            normalized_depth = cv2.normalize(normalized_depth, None, 0, 255, cv2.NORM_MINMAX)
            normalized_depth = normalized_depth.astype(np.uint8)

            # Apply color map
            depth_color = cv2.applyColorMap(normalized_depth, cv2.COLORMAP_JET)

            # Overlay depth map on the original image
            overlay = cv2.addWeighted(img, 0.6, depth_color, 0.4, 0)

            # Display the overlay
            cv2.imshow(window_name, overlay)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            # Optionally, save the visualization
            if save_path:
                cv2.imwrite(save_path, overlay)
                self.logger.info(f"Depth map visualization saved to {save_path}.")

        except Exception as e:
            self.logger.exception(f"Error during depth map visualization: {e}")
            raise
