import logging

import cv2
import numpy as np
from screeninfo import get_monitors  # To get screen resolution

from src.data_loader.dataset import Frame, TrackedObject


class Visualizer:
    """Visualize the frames with tracked objects and a combined depth map using color from both images."""

    def __init__(self):
        """Initialize the Visualizer with appropriate logging configurations and screen size."""
        self.logger = logging.getLogger("autonomous_perception.visualization")
        if not self.logger.hasHandlers():
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        self.selected_track_id = None
        self.track_info = {}

        # Define a color mapping based on object classes
        self.class_color_mapping = {
            "car": (255, 0, 0),  # Blue
            "person": (0, 255, 0),  # Green
            "bicycle": (0, 0, 255),  # Red
            # Add more classes and colors as needed
        }

        # Default color for undefined classes
        self.default_color = (128, 128, 128)  # Gray

        # Get screen resolution
        self.screen_width, self.screen_height = self.get_screen_resolution()
        self.logger.info(f"Detected screen resolution: {self.screen_width}x{self.screen_height}")

        # Define maximum visualization size (e.g., 90% of screen size)
        self.max_width = int(self.screen_width * 0.9)
        self.max_height = int(self.screen_height * 0.9)

    def get_screen_resolution(self):
        """Retrieve the primary monitor's resolution."""
        try:
            monitor = get_monitors()[0]  # Get the first monitor
            return monitor.width, monitor.height
        except Exception as e:
            self.logger.warning(f"Could not get screen resolution: {e}. Using default 1920x1080.")
            return 1920, 1080  # Default resolution

    def reset(self):
        """Reset the visualizer state for a new sequence."""
        self.selected_track_id = None
        self.track_info = {}

    def display(
        self,
        frame: Frame,
        tracked_objects_left: list[TrackedObject],
        tracked_objects_right: list[TrackedObject],
        depth_map: np.ndarray,
    ) -> bool:
        """
        Display the left and right frames with tracked objects and the corresponding depth map.

        Args:
            frame (Frame): The current frame containing left and right images.
            tracked_objects_left (list[TrackedObject]): Tracked objects in the left image.
            tracked_objects_right (list[TrackedObject]): Tracked objects in the right image.
            depth_map (np.ndarray): The depth map derived from the stereo images.

        Returns:
            bool: True to continue displaying, False to exit.

        """
        img_left, img_right = frame.images  # Assuming frame.images returns a tuple (left, right)

        # Draw detections on the left image
        img_left_display = img_left.copy()
        self._draw_tracked_objects(img_left_display, tracked_objects_left, side="Left")

        # Draw detections on the right image
        img_right_display = img_right.copy()
        self._draw_tracked_objects(img_right_display, tracked_objects_right, side="Right")

        # Create a depth heatmap
        depth_heatmap = self._create_depth_heatmap(depth_map)

        # Resize depth map to match image widths
        depth_heatmap_resized = self._resize_depth_map(depth_heatmap, img_left_display.shape[1])

        # Prepare list of images to stack vertically
        images_to_stack = [img_left_display, img_right_display, depth_heatmap_resized]

        # Stack images vertically
        combined_visualization = self._stack_images_vertically(images_to_stack)

        # Calculate scaling factor based on combined image's dimensions
        combined_height, combined_width = combined_visualization.shape[:2]
        scaling_factor = min(
            self.max_width / combined_width,
            self.max_height / combined_height,
            1.0,
        )

        self.logger.debug(
            f"Combined Visualization Size before scaling: Width={combined_width}, Height={combined_height}",
        )
        self.logger.debug(
            f"Scaling factor calculated: {scaling_factor:.4f} "
            f"(Max Width: {self.max_width}, Max Height: {self.max_height})",
        )

        if scaling_factor < 1.0:
            # Resize the combined visualization
            new_width = int(combined_width * scaling_factor)
            new_height = int(combined_height * scaling_factor)
            combined_visualization = cv2.resize(
                combined_visualization,
                (new_width, new_height),
                interpolation=cv2.INTER_AREA,
            )
            self.logger.debug(
                f"Resized combined visualization to ({new_width}, {new_height}) with scaling factor {scaling_factor:.4f}.",
            )
        else:
            self.logger.debug(
                "Combined visualization fits within screen dimensions. No resizing applied.",
            )

        # Display the combined visualization
        cv2.imshow("Autonomous Perception", combined_visualization)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            self.logger.info("Exit key pressed. Closing visualization windows.")
            cv2.destroyAllWindows()
            return False
        return True

    def _draw_tracked_objects(
        self,
        img: np.ndarray,
        tracked_objects: list[TrackedObject],
        side: str,
    ):
        """
        Draw bounding boxes and labels for tracked objects on the given image.

        Args:
            img (np.ndarray): The image on which to draw.
            tracked_objects (list[TrackedObject]): The list of tracked objects.
            side (str): Indicates whether the image is 'Left' or 'Right' for logging purposes.

        """
        for obj in tracked_objects:
            track_id = obj.track_id
            x1, y1, x2, y2 = obj.bbox
            label = obj.label
            position_3d = obj.position_3d
            dimensions = obj.dimensions

            # Assign a color based on the object's class
            color = self.class_color_mapping.get(label, self.default_color)

            # Draw bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            # Prepare text labels
            text = (
                f"ID {track_id} - {label} - "
                f"X: {position_3d[0]:.2f}m, "
                f"Y: {position_3d[1]:.2f}m, "
                f"Z: {position_3d[2]:.2f}m"
            )
            dimension_text = (
                f"H: {dimensions[0]:.2f}m, W: {dimensions[1]:.2f}m, L: {dimensions[2]:.2f}m"
            )
            self.logger.debug(f"{side} Image - {text}")

            # Draw texts
            self._draw_label(img, text, x1, y1 - 30, color)
            self._draw_label(img, dimension_text, x1, y1 - 10, color)

            # Check if the object has a mask
            if hasattr(obj, "mask") and obj.mask is not None:
                mask_coords = np.array(obj.mask)
                if mask_coords.ndim == 2 and mask_coords.shape[1] == 2:
                    # Ensure the coordinates are in integer format
                    mask_polygon = mask_coords.astype(np.int32)
                    # Draw the filled polygon on the image
                    cv2.fillPoly(img, [mask_polygon], color)
                else:
                    self.logger.warning(
                        f"Invalid mask format for track ID {track_id} in {side} image. "
                        f"Expected shape (N, 2), got {obj.mask.shape}",
                    )
            else:
                self.logger.info(f"No mask available for track ID {track_id} in {side} image.")

    def _draw_label(self, img, text, x, y, color):
        """Helper function to draw a label with background."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
        # Adjust rectangle color based on object class color
        rectangle_bgr = color
        cv2.rectangle(
            img,
            (x, y - text_size[1]),
            (x + text_size[0], y + 5),
            rectangle_bgr,
            cv2.FILLED,
        )
        # Determine text color based on rectangle brightness for contrast
        text_color = (0, 0, 0) if self._is_light_color(color) else (255, 255, 255)
        cv2.putText(
            img,
            text,
            (x, y),
            font,
            font_scale,
            text_color,
            thickness,
            cv2.LINE_AA,
        )

    def _is_light_color(self, color):
        """Determine if a BGR color is light based on its luminance."""
        # Calculate luminance using the Rec. 601 luma formula
        luminance = 0.299 * color[2] + 0.587 * color[1] + 0.114 * color[0]
        return luminance > 186  # Threshold for light colors

    def _create_depth_heatmap(self, depth_map: np.ndarray) -> np.ndarray:
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

    def _resize_depth_map(self, depth_heatmap: np.ndarray, target_width: int) -> np.ndarray:
        """
        Resize the depth heatmap to match the target width while maintaining aspect ratio.

        Args:
            depth_heatmap (np.ndarray): The depth heatmap image.
            target_width (int): The desired width.

        Returns:
            np.ndarray: The resized depth heatmap.

        """
        height, width = depth_heatmap.shape[:2]
        if width != target_width:
            scaling_factor = target_width / width
            new_height = int(height * scaling_factor)
            depth_heatmap_resized = cv2.resize(
                depth_heatmap,
                (target_width, new_height),
                interpolation=cv2.INTER_AREA,
            )
            self.logger.debug(
                f"Resized depth heatmap from ({width}, {height}) "
                f"to ({target_width}, {new_height}) to match image width.",
            )
            return depth_heatmap_resized
        return depth_heatmap

    def _stack_images_vertically(self, images: list[np.ndarray]) -> np.ndarray:
        """
        Stack multiple images vertically, ensuring they all fit within the screen.

        Args:
            images (list[np.ndarray]): List of images to stack.

        Returns:
            np.ndarray: The vertically stacked image.

        """
        # Calculate combined height and maximum width
        combined_height = sum(img.shape[0] for img in images)
        max_image_width = max(img.shape[1] for img in images)

        self.logger.debug(
            f"Combined Height before scaling: {combined_height}, Max Width before scaling: {max_image_width}",
        )

        # Calculate scaling factor based on both width and combined height
        scaling_factor = min(
            self.max_width / max_image_width,
            self.max_height / combined_height,
            1.0,
        )

        self.logger.debug(f"Unified Scaling Factor for all images: {scaling_factor:.4f}")

        if scaling_factor < 1.0:
            # Resize all images with the same scaling factor
            resized_images = []
            for img in images:
                new_width = int(img.shape[1] * scaling_factor)
                new_height = int(img.shape[0] * scaling_factor)
                img_resized = cv2.resize(
                    img,
                    (new_width, new_height),
                    interpolation=cv2.INTER_AREA,
                )
                resized_images.append(img_resized)
                self.logger.debug(
                    f"Resized image from ({img.shape[1]}, {img.shape[0]}) "
                    f"to ({new_width}, {new_height}) with scaling factor {scaling_factor:.4f}.",
                )
        else:
            resized_images = images

        # Stack images vertically
        combined_image = np.vstack(resized_images)
        self.logger.debug(
            f"Stacked images vertically to form a combined visualization of size ({combined_image.shape[1]}, {combined_image.shape[0]}).",
        )
        return combined_image
