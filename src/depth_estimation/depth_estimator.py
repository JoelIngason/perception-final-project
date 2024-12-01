import logging
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


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
        mode = cv2.STEREO_SGBM_MODE_SGBM_3WAY

        # Initialize StereoSGBM matcher
        try:
            # self.stereo_matcher = cv2.StereoSGBM.create(
            #    minDisparity=min_disparity,
            #    numDisparities=num_disparities,
            #    blockSize=block_size,
            #    P1=P1,
            #    P2=P2,
            #    disp12MaxDiff=disp12_max_diff,
            #    uniquenessRatio=uniqueness_ratio,
            #    speckleWindowSize=speckle_window_size,
            #    speckleRange=speckle_range,
            #    preFilterCap=pre_filter_cap,
            #    mode=mode,
            # )
            window_size = 3
            min_disp = 6
            num_disp = 112 - min_disp
            # TODO tweak params
            self.stereo_matcher = cv2.StereoSGBM.create(
                minDisparity=min_disp,
                numDisparities=num_disp,
                blockSize=16,
                P1=8 * 3 * window_size**2,
                P2=32 * 3 * window_size**2,
                disp12MaxDiff=1,
                uniquenessRatio=10,
                speckleWindowSize=100,
                speckleRange=32,
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

            if peak_value > 50.0:
                self.logger.warning(f"Large depth value ({peak_value:.2f}m) detected for object.")

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
        image_paths_left: list[str],
        image_paths_right: list[str],
        labels: pd.DataFrame,
        show: bool = False,
    ) -> None:
        """
        Evaluate the depth estimation against ground truth labels.

        Args:
            image_paths_left (List[str]): List of file paths to left images.
            image_paths_right (List[str]): List of file paths to right images.
            labels (pd.DataFrame): Ground truth labels DataFrame.
            show (bool): Whether to display images and plots.

        """
        import matplotlib.pyplot as plt

        errors = []
        disparities_vs_depths = []

        for idx, (left_path, right_path) in enumerate(
            zip(image_paths_left, image_paths_right, strict=False)
        ):
            # Load images
            img_left = cv2.imread(left_path)
            img_right = cv2.imread(right_path)

            if img_left is None or img_right is None:
                self.logger.error(f"Could not read images at index {idx}")
                continue

            # Compute disparity map
            disparity_map = self.compute_disparity_map(img_left, img_right)

            # Compute depth map
            depth_map = self.compute_depth_map(disparity_map, img_left)

            # Get labels for current frame
            current_labels = labels[labels["frame"] == idx]

            frame_errors = []

            for _, row in current_labels.iterrows():
                # Get ROI
                x1, y1 = int(row["bbox_left"]), int(row["bbox_top"])
                x2, y2 = int(row["bbox_right"]), int(row["bbox_bottom"])
                bbox = np.array([x1, y1, x2, y2])

                # Compute estimated depth in the ROI
                estimated_depth, _ = self.compute_object_depth(bbox, depth_map)

                # Ground truth depth (assuming z is depth)
                gt_z = row["z"]

                # Compare estimated depth to ground truth
                if np.isnan(estimated_depth):
                    self.logger.warning(f"No valid depth for object at frame {idx}")
                    continue

                error = np.abs(estimated_depth - gt_z)
                frame_errors.append(error)
                disparities_vs_depths.append([1 / estimated_depth, gt_z])

                if show:
                    # Optionally display images and bounding boxes
                    cv2.rectangle(img_left, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        img_left,
                        f"Est Z: {estimated_depth:.2f}m",
                        (x1, y1 - 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        1,
                    )
                    cv2.putText(
                        img_left,
                        f"GT Z: {gt_z:.2f}m",
                        (x1, y1 - 40),
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
        if np.all(np.isnan(depth_map)):
            self.logger.error("Depth map contains only NaN values. Skipping visualization.")
            return

        min_depth = np.nanmin(depth_map)
        max_depth = np.nanmax(depth_map)
        self.logger.info(f"Depth Map Stats: min={min_depth:.2f} m, max={max_depth:.2f} m")

        # Normalize depth values for visualization
        depth_normalized = cv2.normalize(
            depth_map,
            None,
            alpha=0,
            beta=255,
            norm_type=cv2.NORM_MINMAX,
            dtype=cv2.CV_8U,
        )

        # Apply color mapping
        depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)

        # Resize depth map to match image size if necessary
        if depth_colored.shape[:2] != img.shape[:2]:
            self.logger.warning(
                f"Resizing depth map from {depth_colored.shape[:2]} to {img.shape[:2]}.",
            )
            depth_colored = cv2.resize(
                depth_colored,
                (img.shape[1], img.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

        # Plotting
        fig, ax = plt.subplots(1, 2, figsize=(15, 7))
        ax[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax[0].set_title("Original Image")
        ax[0].axis("off")

        ax[1].imshow(cv2.cvtColor(depth_colored, cv2.COLOR_BGR2RGB))
        ax[1].set_title("Depth Map")
        ax[1].axis("off")

        plt.suptitle(window_name)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            self.logger.info(f"Depth map visualization saved at: {save_path}")

        plt.show()


# The rest of your code remains unchanged, including the StereoCalibrator class
# and the main execution block.

if __name__ == "__main__":
    import yaml

    from src.calibration.stereo_calibration import StereoCalibrator

    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    from pathlib import Path

    # Paths (Update these paths according to your directory structure)
    config_path = "config/config.yaml"
    calibration_file = "data/34759_final_project_rect/calib_cam_to_cam.txt"
    images_left_dir = "data/34759_final_project_rect/seq_01/image_02/data/"
    images_right_dir = "data/34759_final_project_rect/seq_01/image_03/data/"
    labels_file = "data/34759_final_project_rect/seq_01/labels.csv"

    # Load configuration
    if not Path(config_path).is_file():
        raise FileNotFoundError(f"Configuration file not found at {config_path}")
    with open(config_path) as file:
        config = yaml.safe_load(file)

    # Initialize StereoCalibrator and perform calibration
    calibrator = StereoCalibrator(config)
    calib_params = calibrator.calibrate(calibration_file)

    # Load image paths
    image_paths_left = sorted(Path(images_left_dir).glob("*.png"))
    image_paths_right = sorted(Path(images_right_dir).glob("*.png"))
    image_paths_left = [str(p) for p in image_paths_left]
    image_paths_right = [str(p) for p in image_paths_right]

    # Load labels
    labels = pd.read_csv(labels_file)

    # Initialize DepthEstimator
    depth_estimator = DepthEstimator(calib_params, config)

    # Evaluate depth estimation
    depth_estimator.evaluate_depth_estimation(
        image_paths_left,
        image_paths_right,
        labels,
        show=True,
    )
