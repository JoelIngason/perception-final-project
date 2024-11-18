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
    ) -> None:
        """
        Display the frame with tracked objects and their 3D positions,along with a depth heatmap.

        Args:
            frame (Frame): Frame object containing rectified images and metadata.
            tracked_objects (List[TrackedObject]): List of tracked objects with 3D positions.
            disparity_map (np.ndarray): Disparity map of the current frame for depth visualization.

        """
        img_left, img_right = frame.image_left, frame.image_right

        # Create a copy of the left image to draw bounding boxes and labels
        img_display = img_left.copy()

        # Initialize a blank depth map with the same dimensions as the left image
        depth_map = np.zeros_like(img_left[:, :, 0], dtype=np.float32)

        # Populate the depth map based on tracked objects
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

        # Debugging: Log depth_map statistics before normalization
        min_depth = depth_map.min()
        max_depth = depth_map.max()
        mean_depth = depth_map.mean()
        self.logger.debug(
            f"Depth Map Stats - Min: {min_depth:.2f}m, Max: {max_depth:.2f}m, Mean: {mean_depth:.2f}m",
        )

        # Normalize the depth map for visualization
        if max_depth > 0:
            # Define maximum expected depth for normalization (adjust as needed)
            max_depth_visual = 50.0  # meters
            depth_map_clipped = np.clip(depth_map, 0, max_depth_visual)
            depth_map_normalized = (depth_map_clipped / max_depth_visual) * 255.0
            depth_map_normalized = depth_map_normalized.astype(np.uint8)
        else:
            self.logger.warning("All depth values are zero. Creating a uniform depth heatmap.")
            depth_map_normalized = np.zeros_like(depth_map, dtype=np.uint8)

        # Apply a colormap to the normalized depth map to create a heatmap
        depth_heatmap = cv2.applyColorMap(depth_map_normalized, cv2.COLORMAP_JET)

        # Overlay labels on the depth heatmap
        for obj in tracked_objects:
            track_id = obj.track_id
            x, y, w, h = obj.bbox
            x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
            label = obj.label
            position_3d = obj.position_3d  # (X, Y, Z) in meters

            # Prepare label text with 3D position (only Z for depth heatmap)
            text = f"ID {track_id} - {label} - Z: {position_3d[2]:.2f}m"

            # Calculate position for text
            text_position = (
                x1,
                y1 - TEXT_OFFSET if y1 - TEXT_OFFSET > TEXT_OFFSET else y1 + TEXT_OFFSET,
            )

            # Draw label background for better readability
            (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(
                depth_heatmap,
                (text_position[0], text_position[1] - text_height - 5),
                (text_position[0] + text_width, text_position[1] + 5),
                (0, 0, 255),  # Red background for contrast
                cv2.FILLED,
            )

            # Put text on the heatmap
            cv2.putText(
                depth_heatmap,
                text,
                text_position,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),  # White text for contrast
                2,
                cv2.LINE_AA,
            )

        # Combine the 2D image and the depth heatmap vertically
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

    def _get_center(self, bbox: list[int]) -> tuple[float, float]:
        """
        Calculate the center of a bounding box.

        Args:
            bbox (List[int]): Bounding box coordinates [x1, y1, x2, y2].

        Returns:
            Tuple[float, float]: Center coordinates (x, y).

        """
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

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
