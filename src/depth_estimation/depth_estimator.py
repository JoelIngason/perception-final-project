import logging

import cv2
import matplotlib.pyplot as plt
import numpy as np

from src.detection.detection_result import DetectionResult


class DepthEstimator:
    """DepthEstimator class to compute disparity maps, estimate depth from stereo images."""

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

        # Compute focal length and baseline from rectified projection matrices
        try:
            P_rect_00 = calib_params["P_rect_00"]  # Left camera projection matrix
            P_rect_01 = calib_params["P_rect_01"]  # Right camera projection matrix

            # Extract focal length from P_rect_00
            self.focal_length = P_rect_00[0, 0]  # fx from the left camera
            self.logger.info(f"Focal Length (fx): {self.focal_length}")

            # Compute baseline (in meters)
            # P[0, 3] = -fx * baseline
            # Therefore, baseline = -P[0, 3] / fx
            self.baseline = -P_rect_01[0, 3] / self.focal_length
            self.logger.info(f"Baseline (B): {self.baseline} meters")
        except KeyError as e:
            self.logger.exception(f"Missing calibration parameter: {e}")
            raise
        except Exception as e:
            self.logger.exception(f"Error initializing DepthEstimator: {e}")
            raise

        # StereoSGBM parameters (optimized for better depth estimation)
        min_disparity = self.config["stereo"].get("min_disparity", 0)
        num_disparities = self.config["stereo"].get(
            "num_disparities",
            160,
        )  # Must be divisible by 16
        block_size = self.config["stereo"].get("block_size", 7)  # Must be odd

        # Adjusted P1 and P2 for better smoothness
        P1 = 8 * 3 * block_size**2
        P2 = 32 * 3 * block_size**2

        # Fine-tuned matching parameters
        disp12_max_diff = self.config["stereo"].get("disp12_max_diff", 1)
        uniqueness_ratio = self.config["stereo"].get("uniqueness_ratio", 15)
        speckle_window_size = self.config["stereo"].get("speckle_window_size", 100)
        speckle_range = self.config["stereo"].get("speckle_range", 32)
        pre_filter_cap = self.config["stereo"].get("pre_filter_cap", 63)
        mode = cv2.STEREO_SGBM_MODE_SGBM_3WAY  # Higher quality mode

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
            np.ndarray: Refined disparity map.

        """
        try:
            rgb = True  # Assume input images are in RGB format

            # Convert images to grayscale if they are not already
            if len(img_left.shape) == 3 and rgb:
                gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
            else:
                gray_left = img_left

            if len(img_right.shape) == 3 and rgb:
                gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)
            else:
                gray_right = img_right

            # Compute disparity map
            self.logger.info("Computing disparity map...")
            disparity_map = (
                self.stereo_matcher.compute(gray_left, gray_right).astype(np.float32) / 16.0
            )

            # Post-processing: Apply median filter for refinement
            self.logger.info("Refining disparity map with median filter...")
            disparity_map = cv2.medianBlur(disparity_map, 5)

            # Mask invalid disparities by setting them to NaN
            disparity_map[disparity_map <= 0.0] = np.nan

            # Log disparity map statistics
            valid_disparities = disparity_map[~np.isnan(disparity_map)]
            if valid_disparities.size > 0:
                self.logger.debug(
                    f"Disparity Map Stats - Min: {np.nanmin(valid_disparities)}, "
                    f"Max: {np.nanmax(valid_disparities)}, Mean: {np.nanmean(valid_disparities)}",
                )
            else:
                self.logger.warning("No valid disparities found in the disparity map.")

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
            self.logger.info("Computing depth map...")
            depth_map = (self.focal_length * self.baseline) / disparity_map

            # Handle invalid depth values
            depth_map[np.isnan(depth_map) | np.isinf(depth_map) | (depth_map <= 0.0)] = np.nan

            # Log depth map statistics
            valid_depths = depth_map[~np.isnan(depth_map)]
            if valid_depths.size > 0:
                self.logger.debug(
                    f"Depth Map Stats - Min: {np.nanmin(valid_depths)}, "
                    f"Max: {np.nanmax(valid_depths)}, Mean: {np.nanmean(valid_depths)}",
                )
            else:
                self.logger.warning("No valid depths found in the depth map.")

            return depth_map

        except Exception as e:
            self.logger.exception(f"Error computing depth map: {e}")
            raise

    def _compute_object_depth(self, detection: DetectionResult, depth_map: np.ndarray) -> float:
        """Compute the median depth within the object's mask."""
        x, y, w, h = map(int, detection.bbox)
        if detection.mask is None or detection.mask.size == 0:
            self.logger.debug("No mask available for depth computation.")
            return -1.0  # Invalid depth

        # Ensure bounding box is within image boundaries
        img_h, img_w = depth_map.shape
        x_end = min(x + w, img_w)
        y_end = min(y + h, img_h)
        x = max(x, 0)
        y = max(y, 0)
        w = x_end - x
        h = y_end - y

        # Extract the depth ROI corresponding to the bounding box
        depth_roi = depth_map[y : y + h, x : x + w]
        mask_roi = detection.mask[y : y + h, x : x + w].astype(bool)

        if depth_roi.shape != mask_roi.shape:
            self.logger.warning(
                f"Depth ROI and mask size mismatch. Depth ROI: {depth_roi.shape}, Mask ROI: {mask_roi.shape}"
            )
            return -1.0  # Invalid depth

        # Extract depth values where mask is True
        depth_values = depth_roi[mask_roi]
        valid_depths = depth_values[np.isfinite(depth_values) & (depth_values > 0)]

        self.logger.debug(f"Depth Values Count: {valid_depths.size}")

        if valid_depths.size > 0:
            median_depth = np.median(valid_depths)
            self.logger.debug(f"Median Depth for {detection.label}: {median_depth:.2f}m")
            return median_depth.item()
        self.logger.warning(f"No valid depth values for detection {detection.label}")
        return -1.0  # Invalid depth

    def visualize_disparity_map(
        self,
        disparity_map: np.ndarray,
        max_disparity: float = 0.0,
    ) -> None:
        """
        Visualize the disparity map using a color map.

        Args:
            disparity_map (np.ndarray): Disparity map.
            max_disparity (float, optional): Maximum disparity value for normalization. Defaults to 0.0.

        """
        try:
            plt.figure(figsize=(10, 5))
            if max_disparity <= 0.0:
                max_disparity = np.nanmax(disparity_map)
            disparity_display = np.copy(disparity_map)
            disparity_display[np.isnan(disparity_display)] = max_disparity
            plt.imshow(disparity_display, cmap="plasma")
            plt.colorbar(label="Disparity (pixels)")
            plt.title("Disparity Map")
            plt.xlabel("Pixel X")
            plt.ylabel("Pixel Y")
            plt.show()
        except Exception as e:
            self.logger.exception(f"Error visualizing disparity map: {e}")
            raise

    def visualize_depth_map(self, depth_map: np.ndarray, max_depth: float = 50.0) -> None:
        """
        Visualize the depth map using a color map.

        Args:
            depth_map (np.ndarray): Depth map in meters.
            max_depth (float, optional): Maximum depth to display. Defaults to 50.0.

        """
        try:
            plt.figure(figsize=(10, 5))
            depth_display = np.copy(depth_map)
            depth_display[depth_display > max_depth] = max_depth
            depth_display[np.isnan(depth_display)] = 0
            plt.imshow(depth_display, cmap="plasma")
            plt.colorbar(label="Depth (m)")
            plt.title("Depth Map")
            plt.xlabel("Pixel X")
            plt.ylabel("Pixel Y")
            plt.show()
        except Exception as e:
            self.logger.exception(f"Error visualizing depth map: {e}")

    def visualize_point_cloud(
        self,
        depth_map: np.ndarray,
        img_color: np.ndarray,
        max_depth: float = 50.0,
    ) -> None:
        """
        Visualize the depth map as a 3D point cloud using Open3D.

        Args:
            depth_map (np.ndarray): Depth map in meters.
            img_color (np.ndarray): Original color image for coloring the point cloud.
            max_depth (float, optional): Maximum depth to include in the point cloud. Defaults to 50.0.

        """
        try:
            import open3d as o3d
        except ImportError:
            self.logger.error("Open3D is not installed. Point cloud visualization requires Open3D.")
            self.logger.error("Install it using 'pip install open3d' and try again.")
            return

        try:
            # Create mesh grid
            h, w = depth_map.shape
            i_coords = np.arange(w)
            j_coords = np.arange(h)
            i, j = np.meshgrid(i_coords, j_coords)

            # Flatten the arrays
            i = i.flatten()
            j = j.flatten()
            depth = depth_map.flatten()
            colors = img_color.reshape(-1, 3)

            # Filter out points with invalid depth
            mask = (depth > 0) & (depth < max_depth)
            i = i[mask]
            j = j[mask]
            depth = depth[mask]
            colors = colors[mask] / 255.0  # Normalize colors

            # Compute 3D coordinates
            fx = self.calib_params["P_rect_00"][0, 0]
            fy = self.calib_params["P_rect_00"][1, 1]
            cx = self.calib_params["P_rect_00"][0, 2]
            cy = self.calib_params["P_rect_00"][1, 2]

            X = (i - cx) * depth / fx
            Y = (j - cy) * depth / fy
            Z = depth
            points = np.vstack((X, Y, Z)).transpose()

            # Create Open3D point cloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(colors)

            # Visualize
            self.logger.info("Visualizing 3D point cloud...")
            o3d.visualization.draw_geometries([pcd])

        except Exception as e:
            self.logger.exception(f"Error visualizing point cloud: {e}")
            raise
