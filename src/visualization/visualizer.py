import logging
from collections import defaultdict

import cv2
import numpy as np
from screeninfo import get_monitors  # To get screen resolution


def is_light_color(color):
    """
    Determine if a BGR color is light based on its luminance.

    Args:
        color (tuple): BGR color tuple.

    Returns:
        bool: True if the color is light, False otherwise.

    """
    luminance = 0.299 * color[2] + 0.587 * color[1] + 0.114 * color[0]
    return luminance > 186  # Threshold for light colors


class Visualizer:
    """Visualize the frames with tracked objects and a combined depth map using color from both images."""

    def __init__(self, class_color_mapping=None, default_color=(128, 128, 128)):
        """
        Initialize the Visualizer with appropriate logging configurations and screen size.

        Args:
            class_color_mapping (dict, optional): Mapping from class names or IDs to BGR color tuples.
            default_color (tuple, optional): Default color for undefined classes.

        """
        self.logger = logging.getLogger("autonomous_perception.visualization")
        if not self.logger.hasHandlers():
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        self.selected_track_id = None
        self.track_info = {}

        # Define a color mapping based on object classes
        self.class_color_mapping = (
            class_color_mapping
            if class_color_mapping
            else {
                "car": (255, 0, 0),  # Blue
                "person": (0, 255, 0),  # Green
                "bicycle": (0, 0, 255),  # Red
                # Add more classes and colors as needed
            }
        )

        # Default color for undefined classes
        self.default_color = default_color  # Gray

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
        img_left: np.ndarray,
        img_right: np.ndarray,
        tracks_active: list,
        tracks_lost: list,
        names: dict,
        depth_map: np.ndarray,
        conf=True,
        line_width=2,
        font_size=0.5,
        font=cv2.FONT_HERSHEY_SIMPLEX,
        labels=True,
        show=True,
        save=False,
        filename=None,
        color_mode="class",
        centroids=list[list[float]],
    ) -> np.ndarray:
        """
        Display the left and right frames with tracked objects and the corresponding depth map.

        Args:
            img_left (np.ndarray): Original left image as a numpy array.
            img_right (np.ndarray): Annotated left image with ground truth as a numpy array.
            tracks_active (list): List of active tracks (STrack objects).
            tracks_lost (list): List of lost tracks (STrack objects).
            names (dict): Dictionary mapping class IDs to class names.
            depth_map (np.ndarray): Depth map derived from stereo images.
            conf (bool): Whether to display confidence scores.
            line_width (int): Width of bounding box lines.
            font_size (float): Font scale for class labels.
            font (int): Font type for class labels.
            labels (bool): Whether to display class labels.
            show (bool): Whether to display the annotated image.
            save (bool): Whether to save the annotated image.
            filename (str, optional): Name of the file to save the annotated image.
            color_mode (str): Color mode for bounding boxes ("class" or "instance").
            centroids (List[List[float]]): List of centroids for detected objects.

        Returns:
            (np.ndarray): Annotated image as a numpy array.

        """
        assert color_mode in {
            "instance",
            "class",
        }, f"Expected color_mode='instance' or 'class', not {color_mode}."

        # Combine confirmed and lost tracks
        all_tracks = []
        if len(tracks_active) > 0:
            all_tracks.extend(tracks_active)
        if len(tracks_lost) > 0:
            all_tracks.extend(tracks_lost)
        self.logger.debug(f"All tracks: {len(all_tracks)}")

        # Generate a color map for track IDs based on class
        # Red for pedestrians, green for cyclists, blue for vehicles
        # pedastrians: 0, cyclists: 1, vehicles: 2
        if color_mode == "class":
            colors = {
                0: (255, 0, 0),  # Red for class ID 0
                1: (0, 255, 0),  # Green for class ID 1
                2: (0, 0, 255),  # Blue for class ID 2
                # Add more class ID mappings as needed
            }
        else:
            # color everything as red
            colors = defaultdict(lambda: (255, 0, 0))

        annotated_img = img_left.copy()

        for track in all_tracks:
            # Get bounding box in xyzxy format
            bbox = track.xyzxy  # [x1, y1, z, x2, y2]

            # Check if depth is available
            depth = bbox[2] if len(bbox) > 2 else None

            bbox_coords = [bbox[0], bbox[1], bbox[3], bbox[4]]  # [x1, y1, x2, y2]
            bbox_coords = bbox_coords if isinstance(bbox_coords, list) else bbox_coords.tolist()

            # Cap the bbox values to be within the image and not negative
            bbox_coords[0] = min(max(0, int(bbox_coords[0])), annotated_img.shape[1] - 1)
            bbox_coords[1] = min(max(0, int(bbox_coords[1])), annotated_img.shape[0] - 1)
            bbox_coords[2] = max(min(int(bbox_coords[2]), annotated_img.shape[1] - 1), 0)
            bbox_coords[3] = max(min(int(bbox_coords[3]), annotated_img.shape[0] - 1), 0)

            if bbox_coords[0] >= bbox_coords[2] or bbox_coords[1] >= bbox_coords[3]:
                self.logger.debug(f"Invalid bbox for track ID {track.track_id}: {bbox_coords}")
                continue

            # Get class, score, and track ID
            cls_id = int(track.cls) if track.cls is not None else -1
            cls_name = names.get(cls_id, "Unknown") if cls_id != -1 else "Unknown"
            score = float(track.score) if track.score is not None else 0.0
            track_id = track.track_id

            # Define label
            if labels:
                if depth is not None and depth > 0:
                    label = (
                        f"ID:{track_id} {cls_name} {score:.2f} D:{depth:.2f}m"
                        if conf
                        else f"ID:{track_id} {cls_name} D:{depth:.2f}m"
                    )
                else:
                    label = (
                        f"ID:{track_id} {cls_name} {score:.2f}"
                        if conf
                        else f"ID:{track_id} {cls_name}"
                    )
            else:
                label = f"ID:{track_id}" if conf else f"ID:{track_id}"

            # Choose color
            color = colors[cls_id] if cls_id in colors else self.default_color

            # Draw bounding box
            cv2.rectangle(
                annotated_img,
                (bbox_coords[0], bbox_coords[1]),
                (bbox_coords[2], bbox_coords[3]),
                color,
                line_width,
            )

            # Draw label background
            if labels:
                (text_width, text_height), baseline = cv2.getTextSize(label, font, font_size, 1)
                cv2.rectangle(
                    annotated_img,
                    (bbox_coords[0], bbox_coords[1] - text_height - baseline),
                    (bbox_coords[0] + text_width, bbox_coords[1]),
                    color,
                    thickness=cv2.FILLED,
                )
                # Determine text color based on rectangle brightness for contrast
                text_color = (0, 0, 0) if is_light_color(color) else (255, 255, 255)
                cv2.putText(
                    annotated_img,
                    label,
                    (bbox_coords[0], bbox_coords[1] - baseline),
                    font,
                    font_size,
                    text_color,
                    1,
                    cv2.LINE_AA,
                )

        # Create a depth heatmap
        depth_heatmap = self._create_depth_heatmap(depth_map)

        # draw centroids on depth map
        for centroid in centroids:
            cv2.circle(depth_heatmap, (int(centroid[0]), int(centroid[1])), 5, (0, 255, 0), -1)

        # Prepare list of images to stack vertically
        images_to_stack = [annotated_img, img_right, depth_heatmap]

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

        # Save results
        if save:
            if filename is None:
                filename = "annotated_image.jpg"
            cv2.imwrite(filename, combined_visualization)
            self.logger.info(f"Annotated image saved as {filename}")

        # Display the combined visualization
        if show:
            cv2.imshow("Autonomous Perception", combined_visualization)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                self.logger.info("Exit key pressed. Closing visualization windows.")
                cv2.destroyAllWindows()
                return False

        return combined_visualization

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

    def _stack_images_vertically(self, images: list) -> np.ndarray:
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
