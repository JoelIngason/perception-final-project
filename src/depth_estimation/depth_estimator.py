import logging

# Check if src is in the PYTHONPATH
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import zscore


def preprocess_image(image: np.ndarray) -> np.ndarray:
    """
    Apply histogram equalization to improve contrast.

    Args:
        image (np.ndarray): Grayscale image.

    Returns:
        np.ndarray: Preprocessed image.

    """
    if len(image.shape) != 2:
        raise ValueError("Input image for preprocessing must be grayscale.")
    return cv2.equalizeHist(image)


def get_non_overlapping_region(
    disparity: np.ndarray,
    min_disparity: int,
    num_disparities: int,
    width: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Identify non-overlapping regions based on disparity map.

    Args:
        disparity (np.ndarray): Disparity map.
        min_disparity (int): Minimum disparity value.
        num_disparities (int): Number of disparities.
        width (int): Width of the image.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Masks for left and right non-overlapping regions.

    """
    # Create mask for valid disparity (within min and max disparity range)
    max_disparity = min_disparity + num_disparities
    valid_mask = (disparity >= min_disparity) & (disparity <= max_disparity)

    # Identify the overlap region (valid disparity region)
    if np.any(valid_mask):
        overlap_start = np.argmax(valid_mask.any(axis=0))
        # To handle cases where valid_mask is all False in reversed
        reversed_valid_mask = valid_mask[:, ::-1]
        if np.any(reversed_valid_mask):
            overlap_end = width - np.argmax(reversed_valid_mask.any(axis=0))
        else:
            overlap_end = width
    else:
        overlap_start, overlap_end = 0, width

    # Compute the non-overlapping regions
    left_non_overlap = np.ones_like(valid_mask, dtype=np.uint8)
    left_non_overlap[:, :overlap_start] = 0  # Non-overlapping part of the left image

    right_non_overlap = np.ones_like(valid_mask, dtype=np.uint8)
    right_non_overlap[:, overlap_end:] = 0  # Non-overlapping part of the right image

    return left_non_overlap, right_non_overlap


def compute_scale_and_shift(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute scale and shift factors to align predicted depth with target.

    Args:
        prediction (torch.Tensor): Predicted depth.
        target (torch.Tensor): Target depth.
        mask (torch.Tensor): Mask indicating valid regions.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Scale and shift tensors.

    """
    a_00 = torch.sum(mask * prediction * prediction, dim=(1, 2))
    a_01 = torch.sum(mask * prediction, dim=(1, 2))
    a_11 = torch.sum(mask, dim=(1, 2))

    b_0 = torch.sum(mask * prediction * target, dim=(1, 2))
    b_1 = torch.sum(mask * target, dim=(1, 2))

    det = a_00 * a_11 - a_01 * a_01
    valid = det > 0

    scale = torch.zeros_like(b_0)
    shift = torch.zeros_like(b_1)

    scale[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
    shift[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]

    return scale, shift


def to_depth(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    depth_cap: float = 80.0,
) -> np.ndarray:
    """
    Transform predicted disparity to aligned depth.

    Args:
        prediction (torch.Tensor): Predicted depth.
        target (torch.Tensor): Target depth.
        mask (torch.Tensor): Mask indicating valid regions.
        depth_cap (float, optional): Maximum depth value. Defaults to 80.0.

    Returns:
        np.ndarray: Aligned depth map.

    """
    # Avoid division by zero
    target_disparity = torch.zeros_like(target)
    target_disparity[mask == 1] = 1.0 / target[mask == 1]

    scale, shift = compute_scale_and_shift(prediction, target_disparity, mask)
    prediction_aligned = scale.view(-1, 1, 1) * prediction + shift.view(-1, 1, 1)

    disparity_cap = 1.0 / depth_cap
    prediction_aligned[prediction_aligned < disparity_cap] = disparity_cap

    prediction_depth = 1.0 / prediction_aligned

    # Clamp the depth values to the specified cap
    prediction_depth = prediction_depth.clamp(max=depth_cap)

    return prediction_depth.cpu().numpy()


class DepthEstimator:
    """DepthEstimator class to compute disparity maps and estimate depth from stereo images."""

    def __init__(
        self,
        calib_params: dict[str, np.ndarray],
        config: dict[str, Any],
        image_size: tuple[int, int] = (1280, 720),
    ):
        """
        Initialize the DepthEstimator with calibration parameters and configuration.

        Args:
            calib_params (Dict[str, np.ndarray]): Stereo calibration parameters.
            config (Dict[str, Any]): Configuration parameters.
            image_size (Tuple[int, int], optional): Image size for disparity map computation.

        """
        self.logger = logging.getLogger("autonomous_perception.depth_estimation")
        self.calib_params = calib_params
        self.config = config
        self.image_size = image_size

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
        P1 = stereo_config.get("P1", 8 * 3 * block_size**2)
        P2 = stereo_config.get("P2", 32 * 3 * block_size**2)
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
            # Check if disparity map has valid values
            if np.all(disparity_map <= 0):
                self.logger.error("Disparity map contains only invalid values.")
                raise ValueError("Invalid disparity map computed.")

            # Post-process disparity map
            disparity_map[disparity_map <= 0.0] = np.nan  # Mask invalid disparities

            self.logger.debug("Disparity map computed successfully.")
            self.logger.debug(
                f"Disparity Map Stats: min={np.nanmin(disparity_map)}, max={np.nanmax(disparity_map)}, mean={np.nanmean(disparity_map)}",
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
            # Compute depth using the formula:
            # Depth = (focal_length * baseline) / (disparity * image_width + epsilon)
            with np.errstate(divide="ignore", invalid="ignore"):
                depth_map = (self.focal_length * self.baseline) / (disparity_map + 1e-6)
                depth_map[disparity_map <= 0.0] = np.nan  # Mask invalid disparities
                depth_map[depth_map > 80.0] = np.nan  # Cap depth values at 80m

            self.logger.debug("Depth map computed successfully.")

            # Check if depth_map contains valid values
            if np.all(np.isnan(depth_map)):
                self.logger.error("Depth map contains only NaN values after computation.")
                raise ValueError("Invalid depth map computed.")

            depth_map = self.refine_depth_map(depth_map)
            return depth_map

        except Exception as e:
            self.logger.exception(f"Error computing depth map: {e}")
            raise

    def compute_object_depth(
        self,
        bbox: np.ndarray,
        depth_map: np.ndarray,
    ) -> tuple[float, float]:
        """
        Compute the median depth within the object's mask, excluding extreme outliers.

        Uses z-score filtering to exclude depth values that deviate significantly from the mean.

        Args:
            bbox (np.ndarray): Bounding box coordinates in the format [x1, y1, x2, y2].
            depth_map (np.ndarray): Depth map in meters.

        Returns:
            Tuple[float, float]: Median depth and object depth range in meters.

        """
        # Extract the object mask from the depth map
        x, y, w, h = bbox
        # Find the center of the bounding box and set the mask
        # as 5 pixels around the center to avoid noise
        center_x = x + w // 2
        center_y = y + h // 2

        # Ensure the mask is within the depth map boundaries
        mask_x_start = int(max(center_x - 5, 0))
        mask_x_end = int(min(center_x + 5, depth_map.shape[1]))
        mask_y_start = int(max(center_y - 5, 0))
        mask_y_end = int(min(center_y + 5, depth_map.shape[0]))

        mask = depth_map[
            mask_y_start:mask_y_end,
            mask_x_start:mask_x_end,
        ]

        # Flatten the mask and remove NaN values
        valid_depths = mask[~np.isnan(mask)]

        # Calculate z-scores for the depth values
        z_scores = zscore(valid_depths)

        # Exclude outliers with a z-score threshold (e.g., |z| > 2.5)
        z_threshold = self.config.get("stereo", {}).get("z_threshold", 2.5)
        filtered_depths = valid_depths[np.abs(z_scores) <= z_threshold]

        # Make sure there are valid depth values
        if len(filtered_depths) == 0:
            self.logger.warning("No valid depth values found in the object mask.")
            return (np.nan, np.nan)
        # Calculate the median of the filtered depth values
        median_depth = float(np.median(filtered_depths))

        # Calculate the depth range of the object (difference between max and min filtered depth)
        object_depth = float(np.max(filtered_depths) - np.min(filtered_depths))
        if object_depth > 10.0:
            self.logger.warning(
                f"Large depth range ({object_depth:.2f}m) detected for object .",
            )
        return (median_depth, object_depth)

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
            # Replace NaNs with zero for filtering
            nan_mask = np.isnan(refined_depth)
            refined_depth[nan_mask] = 0.0

            # Apply bilateral filter (Note: bilateralFilter does not handle NaNs)
            refined_depth = cv2.bilateralFilter(
                refined_depth.astype(np.float32),
                d=9,
                sigmaColor=75,
                sigmaSpace=75,
            )

            # Restore NaNs
            refined_depth[nan_mask] = np.nan
            self.logger.debug("Bilateral filtering applied to depth map.")

            # 2. Hole Filling using Inpainting
            # Create mask of invalid depth values
            invalid_mask = np.isnan(refined_depth).astype(np.uint8) * 255
            if np.any(invalid_mask):
                # Inpaint using Telea's method
                # OpenCV inpaint requires 8-bit or 1-channel images; replace NaNs with zero temporarily
                inpaint_input = np.where(np.isnan(refined_depth), 0, refined_depth).astype(
                    np.float32,
                )
                inpainted_depth = cv2.inpaint(
                    inpaint_input,
                    invalid_mask,
                    inpaintRadius=3,
                    flags=cv2.INPAINT_TELEA,
                )

                # Restore NaNs where inpainting was not possible
                inpainted_depth[invalid_mask == 255] = np.nan
                refined_depth = inpainted_depth
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
        Visualize the depth map alongside the original image using Matplotlib.

        Args:
            img (np.ndarray): Original rectified left image.
            depth_map (np.ndarray): Refined depth map in meters.
            window_name (str, optional): Title for the plot.
            save_path (str, optional): Path to save the visualized depth map image.

        """
        # Check if depth_map contains valid values
        if np.all(np.isnan(depth_map)):
            self.logger.error("Depth map contains only NaN values. Skipping visualization.")
            return

        min_depth = np.nanmin(depth_map)
        max_depth = np.nanmax(depth_map)
        print(f"Min depth: {min_depth:.2f} m, Max depth: {max_depth:.2f} m")

        # Normalize depth values for visualization
        depth_map = cv2.normalize(
            depth_map,
            None,
            alpha=0,
            beta=255,
            norm_type=cv2.NORM_MINMAX,
        )  # type: ignore

        # Create a color map for depth visualization
        depth_map_color = cv2.applyColorMap(depth_map, cv2.COLORMAP_JET)

        # Resize the depth map to match the image size
        depth_map_resized = cv2.resize(
            depth_map_color,
            (img.shape[1], img.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

        # Visualize the depth map alongside the original image
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        ax[0].imshow(img)
        ax[0].set_title("Original Image")
        ax[0].axis("off")

        ax[1].imshow(depth_map_resized)
        ax[1].set_title("Depth Map")
        ax[1].axis("off")

        plt.suptitle(window_name)
        plt.tight_layout()

        plt.show()

        if save_path:
            plt.savefig(save_path)
            self.logger.info(f"Depth map visualization saved at: {save_path}")


if __name__ == "__main__":
    # Example usage of DepthEstimator
    import yaml

    from src.calibration.stereo_calibration import StereoCalibrator

    config_path = "../../config/config.yaml"

    # Load configuration
    with open(config_path) as file:
        config = yaml.safe_load(file)
    # Load stereo calibration parameters
    calibration_file = "../../data/34759_final_project_rect/calib_cam_to_cam.txt"
    calib_params = StereoCalibrator(config).calibrate(calibration_file)
    # Load stereo images
    img_left = cv2.imread("../../data/34759_final_project_rect/seq_01/image_02/data/000000.png")
    img_right = cv2.imread("../../data/34759_final_project_rect/seq_01/image_03/data/000000.png")
    image_size = (img_left.shape[1], img_left.shape[0])
    print(f"Image size: {image_size}")

    # Initialize DepthEstimator
    depth_estimator = DepthEstimator(calib_params, config, image_size)

    if img_left.shape[2] == 3:
        img_left = cv2.cvtColor(img_left, cv2.COLOR_RGB2BGR)
    if img_right.shape[2] == 3:
        img_right = cv2.cvtColor(img_right, cv2.COLOR_RGB2BGR)
    if img_left is None or img_right is None:
        raise FileNotFoundError("Input images not found.")
    # Compute disparity map
    disparity_map = depth_estimator.compute_disparity_map(img_left, img_right)

    # Compute depth map
    depth_map = depth_estimator.compute_depth_map(disparity_map)

    # Visualize depth map
    depth_estimator.visualize_depth_map(
        img_left,
        depth_map,
        window_name="Depth Visualization",
        save_path="depth_map.png",
    )

    # Refine depth map
    refined_depth = depth_estimator.refine_depth_map(depth_map)
    depth_estimator.visualize_depth_map(
        img_left,
        refined_depth,
        window_name="Refined Depth Visualization",
        save_path="refined_depth_map.png",
    )
