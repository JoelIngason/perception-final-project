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

        # Initialize track visualization parameters
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
        """
        Display the frame with tracked objects and their 3D positions, along with a depth heatmap.

        Args:
            frame (Frame): Frame object containing rectified images and metadata.
            tracked_objects (List[TrackedObject]): List of tracked objects with 3D positions.
            depth_map (np.ndarray): Depth map of the current frame for depth visualization.

        Returns:
            bool: True if the visualization should continue, False if exit is requested.

        """
        img_left, _ = frame.images

        # Create a copy of the left image to draw bounding boxes and labels
        img_display = img_left.copy()

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
        cv2.imshow("Autonomous Perception - 2D and Depth Visualization", combined_visualization)

        # Handle keyboard inputs for interactivity
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            self.logger.info("Exit key pressed. Closing visualization windows.")
            cv2.destroyAllWindows()
            return False
        if key == ord("t"):
            # Toggle tracking information display
            self.logger.info("Track information display toggled.")
            self.display_track_info(tracked_objects)
        elif key >= ord("0") and key <= ord("9"):
            # Select track ID based on number key pressed
            self.selected_track_id = int(chr(key))
            self.logger.info(f"Selected Track ID: {self.selected_track_id}")
            self.display_track_info(tracked_objects)
        return True

    def normalize_depth_map(self, depth_map: np.ndarray) -> np.ndarray:
        """
        Normalize the depth map for visualization.

        Args:
            depth_map (np.ndarray): Depth map with depth in meters.

        Returns:
            np.ndarray: Normalized depth map as an 8-bit image.

        """
        # Set invalid depth values to a maximum value for visualization purposes
        max_depth = 50.0  # Maximum depth in meters for visualization
        depth_map_vis = np.copy(depth_map)
        depth_map_vis[depth_map_vis > max_depth] = max_depth
        depth_map_vis[depth_map_vis <= 0] = max_depth

        # Normalize depth map for visualization
        normalized_depth_map = cv2.normalize(
            depth_map_vis,
            None,
            alpha=0,
            beta=255,
            norm_type=cv2.NORM_MINMAX,
            dtype=cv2.CV_8U,
        )
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
