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
        self.selected_track_id = None
        self.track_info = {}

    def reset(self):
        """Reset the visualizer state for a new sequence."""
        self.selected_track_id = None
        self.track_info = {}

    def display(
        self,
        frame: Frame,
        tracked_objects: list[TrackedObject],
        depth_map: np.ndarray,
    ) -> bool:
        """Display the frame with tracked objects and their 3D positions, along with a depth heatmap."""
        img_left, _ = frame.images
        img_display = img_left.copy()

        for obj in tracked_objects:
            track_id = obj.track_id
            x, y, w, h = obj.bbox
            x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
            label = obj.label
            position_3d = obj.position_3d
            dimensions = obj.dimensions

            # Draw oriented bounding box if available
            if hasattr(obj, "oriented_bbox"):
                self._draw_oriented_bbox(img_display, obj.oriented_bbox, obj.label)
            else:
                # Draw regular bounding box
                cv2.rectangle(img_display, (x1, y1), (x2, y2), (0, 255, 0), 2)

            cv2.rectangle(img_display, (x1, y1), (x2, y2), (0, 255, 0), 2)
            text = (
                f"ID {track_id} - {label} - "
                f"X: {position_3d[0]:.2f}m, "
                f"Y: {position_3d[1]:.2f}m, "
                f"Z: {position_3d[2]:.2f}m"
            )
            dimension_text = (
                f"H: {dimensions[0]:.2f}m, W: {dimensions[1]:.2f}m, L: {dimensions[2]:.2f}m"
            )
            # Draw texts
            self._draw_label(img_display, text, x1, y1 - 30)
            self._draw_label(img_display, dimension_text, x1, y1 - 10)

        depth_heatmap = self._create_depth_heatmap(depth_map)
        combined_visualization = self._stack_images_vertically(img_display, depth_heatmap)
        cv2.imshow("Autonomous Perception", combined_visualization)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            self.logger.info("Exit key pressed. Closing visualization windows.")
            cv2.destroyAllWindows()
            return False
        return True

    def _draw_oriented_bbox(self, image, bbox, label):
        """Draw an oriented bounding box on the image."""
        # bbox is expected to be a list of four points (x, y)
        if not bbox or len(bbox) == 0:
            return
        pts = np.array(bbox, np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(image, [pts], True, (0, 255, 0), 2)
        # Put label
        x, y = pts[0][0]
        cv2.putText(
            image,
            label,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    def _draw_label(self, img, text, x, y):
        """Helper function to draw a label with background."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
        cv2.rectangle(
            img,
            (x, y - text_size[1]),
            (x + text_size[0], y + 5),
            (0, 255, 0),
            cv2.FILLED,
        )
        cv2.putText(
            img,
            text,
            (x, y),
            font,
            font_scale,
            (0, 0, 0),
            thickness,
            cv2.LINE_AA,
        )

    def _create_depth_heatmap(self, depth_map):
        """Create a heatmap from the depth map."""
        depth_map_normalized = self.normalize_depth_map(depth_map)
        depth_heatmap = cv2.applyColorMap(depth_map_normalized, cv2.COLORMAP_JET)
        return depth_heatmap

    def normalize_depth_map(self, depth_map: np.ndarray) -> np.ndarray:
        """
        Normalize and invert the depth map for visualization.

        Args:
            depth_map (np.ndarray): Depth map with depth in meters.

        Returns:
            np.ndarray: Inverted and normalized depth map as an 8-bit image.

        """
        # Define the minimum and maximum depth values for visualization
        min_depth = 0.001  # Minimum depth in meters to avoid division by zero and noise
        max_depth = 100.0  # Maximum depth in meters for visualization

        depth_map_vis = np.copy(depth_map)

        # Set invalid depth values to zero
        invalid_mask = (
            (depth_map_vis <= min_depth) | (depth_map_vis > max_depth) | np.isnan(depth_map_vis)
        )
        depth_map_vis[invalid_mask] = 0

        # Invert depth values for visualization
        # Closer objects (smaller depth values) will have higher intensity values
        valid_mask = depth_map_vis > 0
        depth_values = depth_map_vis[valid_mask]

        normalized_depth_map = np.zeros_like(depth_map_vis, dtype=np.uint8)

        if depth_values.size > 0:
            # Invert depth values
            inverted_depth_values = max_depth - depth_values

            # Normalize the inverted depth values to 0-255
            depth_min = inverted_depth_values.min()
            depth_max = inverted_depth_values.max()
            self.logger.debug(f"Depth map inversion: min={depth_min}, max={depth_max}")

            # Avoid division by zero
            if depth_max > depth_min:
                normalized_values = (
                    (inverted_depth_values - depth_min) / (depth_max - depth_min) * 255
                ).astype(np.uint8)
            else:
                normalized_values = np.zeros_like(inverted_depth_values, dtype=np.uint8)
            normalized_depth_map[valid_mask] = normalized_values
        else:
            self.logger.warning("No valid depth values found for normalization.")

        return normalized_depth_map

    def _stack_images_vertically(self, img_top: np.ndarray, img_bottom: np.ndarray) -> np.ndarray:
        """
        Stack two images vertically, resizing them to have the same width.

        Args:
            img_top (np.ndarray): Top image.
            img_bottom (np.ndarray): Bottom image.

        Returns:
            np.ndarray: Combined image stacked vertically.

        """
        height_top, width_top = img_top.shape[:2]
        height_bottom, width_bottom = img_bottom.shape[:2]
        if width_top != width_bottom:
            scaling_factor = width_top / width_bottom
            new_height = int(height_bottom * scaling_factor)
            img_bottom_resized = cv2.resize(
                img_bottom,
                (width_top, new_height),
                interpolation=cv2.INTER_AREA,
            )
            self.logger.info(
                f"Resized bottom image from ({width_bottom}, {height_bottom}) "
                f"to ({width_top}, {new_height}) to match top image width.",
            )
        else:
            img_bottom_resized = img_bottom

        combined_image = np.vstack((img_top, img_bottom_resized))
        return combined_image

    def display_track_info(self, tracked_objects: list[TrackedObject]) -> None:
        """Display detailed information about tracked objects."""
        if self.selected_track_id is not None:
            for obj in tracked_objects:
                if obj.track_id == self.selected_track_id:
                    info = (
                        f"Track ID: {obj.track_id}\n"
                        f"Label: {obj.label}\n"
                        f"3D Position: X={obj.position_3d[0]:.2f}m, "
                        f"Y={obj.position_3d[1]:.2f}m, Z={obj.position_3d[2]:.2f}m"
                    )
                    self.logger.info(info)
                    break
            else:
                self.logger.warning(f"Track ID {self.selected_track_id} not found.")
        else:
            self.logger.info("No Track ID selected.")
