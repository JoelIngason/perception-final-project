import logging

import cv2
import numpy as np

from src.data_loader.dataset import Frame, TrackedObject


class Visualizer:
    """Visualize the frame with tracked objects and depth information."""

    def __init__(self):
        """Initialize the Visualizer with appropriate logging configurations."""
        self.logger = logging.getLogger("autonomous_perception.visualization")
        self.logger.setLevel(logging.INFO)
        if not self.logger.hasHandlers():
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def display(
        self,
        frame: Frame,
        tracked_objects: list[TrackedObject],
        disparity_map: np.ndarray,
        calib_params: dict[str, np.ndarray],
    ) -> None:
        """
        Display the frame with tracked objects and their 3D positions, along with a depth heatmap.

        Args:
            frame (Frame): Frame object containing rectified images and metadata.
            tracked_objects (List[TrackedObject]): List of tracked objects with 3D positions.
            disparity_map (np.ndarray): Disparity map of the current frame for depth visualization.
            calib_params (Dict[str, np.ndarray]): Dictionary containing calibration parameters.

        """
        img_left, img_right = frame.images

        # Create a copy of the left image to draw bounding boxes and labels
        img_display = img_left.copy()

        # Compute the full depth map from the disparity map
        P_rect_02 = calib_params["P_rect_02"]  # 3x4 matrix
        P_rect_03 = calib_params["P_rect_03"]  # 3x4 matrix

        self.focal_length = P_rect_02[0, 0]  # fx from rectified left projection matrix
        self.logger.info(f"Focal length set to {self.focal_length}")

        # Compute baseline from projection matrices
        Tx_left = P_rect_02[0, 3]
        Tx_right = P_rect_03[0, 3]
        self.baseline = np.abs(Tx_right - Tx_left) / self.focal_length
        depth_map = self.compute_depth_map(
            disparity_map,
            self.focal_length,
            self.baseline,
        )

        # Debugging: Log depth_map statistics before normalization
        min_depth = np.min(depth_map[np.isfinite(depth_map) & (depth_map > 0)])
        max_depth = np.max(depth_map[np.isfinite(depth_map) & (depth_map > 0)])
        mean_depth = np.mean(depth_map[np.isfinite(depth_map) & (depth_map > 0)])
        self.logger.debug(
            f"Depth Map Stats - Min: {min_depth:.2f}m, Max: {max_depth:.2f}m, Mean: {mean_depth:.2f}m",
        )

        # Normalize the depth map for visualization
        depth_map_normalized = self.normalize_depth_map(depth_map)

        # Apply a colormap to the normalized depth map to create a heatmap
        depth_heatmap = cv2.applyColorMap(depth_map_normalized, cv2.COLORMAP_JET)

        # Draw tracked objects on the display image
        for obj in tracked_objects:
            track_id = obj.track_id
            x, y, w, h = obj.bbox
            x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
            label = obj.label
            position_3d = obj.position_3d  # (X, Y, Z) in meters

            # Draw bounding box on the display image
            cv2.rectangle(img_display, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Prepare label text with 3D position
            text = (
                f"ID {track_id} - {label} - "
                f"X: {position_3d[0]:.2f}m, "
                f"Y: {position_3d[1]:.2f}m, "
                f"Z: {position_3d[2]:.2f}m"
            )

            # Calculate position for text to avoid overlapping with bounding box
            TEXT_OFFSET = 10
            text_position = (
                x1,
                y1 - TEXT_OFFSET if y1 - TEXT_OFFSET > TEXT_OFFSET else y1 + TEXT_OFFSET,
            )

            # Draw label background for better readability
            (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(
                img_display,
                (text_position[0], text_position[1] - text_height - 5),
                (text_position[0] + text_width, text_position[1] + 5),
                (0, 255, 0),
                cv2.FILLED,
            )

            # Put text on the image
            cv2.putText(
                img_display,
                text,
                text_position,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),  # Black text for contrast
                2,
                cv2.LINE_AA,
            )

        # Overlay the depth heatmap and the display image
        combined_visualization = self._stack_images_vertically(img_display, depth_heatmap)

        # Display the combined visualization
        cv2.imshow(
            "Autonomous Perception - 2D and Depth Visualization",
            combined_visualization,
        )

        # Wait for a short period; allow exit with 'q' key
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            self.logger.info("Exit key pressed. Closing visualization window.")
            cv2.destroyAllWindows()

    def compute_depth_map(
        self,
        disparity_map: np.ndarray,
        focal_length: float,
        baseline: float,
    ) -> np.ndarray:
        """
        Compute the depth map from disparity map using the formula depth = f * B / disparity.

        Args:
            disparity_map (np.ndarray): Disparity map.
            focal_length (float): Focal length (in pixels).
            baseline (float): Baseline (in meters).

        Returns:
            np.ndarray: Depth map with depth in meters.

        """
        with np.errstate(
            divide="ignore",
            invalid="ignore",
        ):  # Handle division by zero and invalid values
            depth_map = (focal_length * baseline) / disparity_map
            depth_map[disparity_map <= 0] = 0  # Assign zero depth where disparity is invalid
        return depth_map

    def normalize_depth_map(self, depth_map: np.ndarray) -> np.ndarray:
        """
        Normalize the depth map for visualization.

        Args:
            depth_map (np.ndarray): Depth map with depth in meters.

        Returns:
            np.ndarray: Normalized depth map as an 8-bit image.

        """
        # Mask invalid depth values
        valid_mask = depth_map > 0
        if np.any(valid_mask):
            min_val = np.min(depth_map[valid_mask])
            max_val = np.max(depth_map[valid_mask])
            # Avoid division by zero if min_val == max_val
            if max_val - min_val > 1e-3:
                depth_map_normalized = np.zeros_like(depth_map, dtype=np.uint8)
                depth_map_normalized[valid_mask] = np.clip(
                    255 * (depth_map[valid_mask] - min_val) / (max_val - min_val),
                    0,
                    255,
                ).astype(np.uint8)
            else:
                depth_map_normalized = np.zeros_like(depth_map, dtype=np.uint8)
        else:
            self.logger.warning("No valid depth values found. Depth heatmap will be all zeros.")
            depth_map_normalized = np.zeros_like(depth_map, dtype=np.uint8)
        return depth_map_normalized

    def _stack_images_vertically(self, img_top: np.ndarray, img_bottom: np.ndarray) -> np.ndarray:
        """
        Stack two images vertically, resizing them to have the same width.

        Args:
            img_top (np.ndarray): Top image.
            img_bottom (np.ndarray): Bottom image.

        Returns:
            np.ndarray: Combined image stacked vertically.

        """
        # Ensure both images have the same width
        height_top, width_top = img_top.shape[:2]
        height_bottom, width_bottom = img_bottom.shape[:2]

        if width_top != width_bottom:
            # Resize bottom image to match top image's width
            scaling_factor = width_top / width_bottom
            new_height = int(height_bottom * scaling_factor)
            img_bottom_resized = cv2.resize(
                img_bottom,
                (width_top, new_height),
                interpolation=cv2.INTER_AREA,
            )
            self.logger.debug(
                f"Resized bottom image from ({width_bottom}, {height_bottom}) "
                f"to ({width_top}, {new_height}) to match top image width.",
            )
        else:
            img_bottom_resized = img_bottom

        # Stack images vertically
        combined_image = np.vstack((img_top, img_bottom_resized))
        return combined_image  # noqa: RET504
