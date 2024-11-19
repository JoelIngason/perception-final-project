import logging
import threading

import cv2
import numpy as np
import open3d as o3d

from src.data_loader.dataset import Frame, TrackedObject


class Visualizer:
    """Visualize the frame with tracked objects and depth information."""

    def __init__(self, calib_params: dict[str, np.ndarray]) -> None:
        """
        Initialize the Visualizer with appropriate logging configurations.

        Args:
            calib_params (Dict[str, np.ndarray]): Stereo calibration parameters.

        """
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

        # Camera intrinsic parameters
        P_rect_02 = calib_params["P_rect_02"]  # 3x4 matrix
        P_rect_03 = calib_params["P_rect_03"]  # 3x4 matrix
        self.focal_length = P_rect_02[0, 0]  # fx from rectified left projection matrix

        # Compute baseline
        Tx_left = P_rect_02[0, 3] / -self.focal_length  # In meters
        Tx_right = P_rect_03[0, 3] / -self.focal_length  # In meters
        self.baseline = Tx_right - Tx_left  # Positive value

        self.cx = P_rect_02[0, 2]  # Principal point x-coordinate
        self.cy = P_rect_02[1, 2]  # Principal point y-coordinate

        # Synchronization lock
        self.lock = threading.Lock()

        # Open3D visualization setup
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name="Point Cloud Visualization", width=960, height=540)
        self.pcd = o3d.geometry.PointCloud()
        self.is_first_frame = True

        # Start Open3D visualization in a separate thread
        self.open3d_thread = threading.Thread(target=self._open3d_visualization_loop, daemon=True)
        self.open3d_thread.start()

    def reset(self):
        """Reset the visualizer state for a new sequence."""
        self.selected_track_id = None
        self.track_info = {}
        self.is_first_frame = True
        with self.lock:
            self.pcd.clear()
            self.vis.clear_geometries()
            self.vis.add_geometry(self.pcd)
            self.vis.reset_view_point(True)

    def close(self):
        """Close the Open3D visualization window."""
        self.vis.destroy_window()

    def display(
        self,
        frame: Frame,
        tracked_objects: list[TrackedObject],
        depth_map: np.ndarray,
        color_image: np.ndarray,
    ) -> bool:
        """
        Display the frame with tracked objects and their 3D positions.

        Args:
            frame (Frame): Frame object containing rectified images and metadata.
            tracked_objects (List[TrackedObject]): List of tracked objects with 3D positions.
            depth_map (np.ndarray): Depth map of the current frame.
            color_image (np.ndarray): The corresponding color image for coloring the point cloud.

        Returns:
            bool: True if the visualization should continue, False if exit is requested.

        """
        # Create a copy of the color image to draw bounding boxes and labels
        img_display = color_image.copy()

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

        # Display the image with bounding boxes
        cv2.imshow("Autonomous Perception - 2D Visualization", img_display)

        # Generate point cloud from depth map
        points, colors = self.generate_point_cloud(depth_map, color_image)

        # Update point cloud data in a thread-safe manner
        with self.lock:
            self.pcd.points = o3d.utility.Vector3dVector(points)
            self.pcd.colors = o3d.utility.Vector3dVector(colors)

        # Handle keyboard inputs for interactivity
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            self.logger.info("Exit key pressed. Closing visualization windows.")
            cv2.destroyAllWindows()
            self.close()
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
        return True  # Continue visualization

    def _open3d_visualization_loop(self):
        """Run the Open3D visualization loop in a separate thread."""
        while True:
            with self.lock:
                self.vis.update_geometry(self.pcd)
            self.vis.poll_events()
            self.vis.update_renderer()

    def generate_point_cloud(
        self,
        depth_map: np.ndarray,
        color_image: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate a point cloud from the depth map and color image.

        Args:
            depth_map (np.ndarray): Depth map with depth in meters.
            color_image (np.ndarray): Corresponding color image.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Tuple of points (Nx3) and colors (Nx3).

        """
        # Get image dimensions
        height, width = depth_map.shape

        # Create meshgrid of pixel coordinates
        u = np.arange(width)
        v = np.arange(height)
        u_grid, v_grid = np.meshgrid(u, v)

        # Flatten the arrays
        u_grid = u_grid.flatten()
        v_grid = v_grid.flatten()
        depth = depth_map.flatten()

        # Filter out invalid depth values
        valid = depth > 0
        u_valid = u_grid[valid]
        v_valid = v_grid[valid]
        depth_valid = depth[valid]

        # Compute 3D points
        x = (u_valid - self.cx) * depth_valid / self.focal_length
        y = (v_valid - self.cy) * depth_valid / self.focal_length
        z = depth_valid

        points = np.vstack((x, y, z)).transpose()

        # Get colors
        color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)  # Convert to RGB
        colors = color_image.reshape(-1, 3)[valid] / 255.0  # Normalize to [0,1]

        return points, colors

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
