import logging

import cv2
import numpy as np

# Import Frame and TrackedObject classes
from src.data_loader.dataset import Frame
from src.detection.object_detector import TrackedObject


class Visualizer:
    def __init__(self):
        """
        Initialize the Visualizer with appropriate logging configurations.
        """
        self.logger = logging.getLogger("autonomous_perception.visualization")

        # Configure logger if not already configured
        if not self.logger.hasHandlers():
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    def display(self, frame: Frame, tracked_objects: list[TrackedObject]):
        """
        Display the frame with tracked objects and their 3D positions.

        Args:
            frame (Frame): Frame object containing rectified images and metadata.
            tracked_objects (List[TrackedObject]): List of tracked objects with 3D positions.

        """
        # Unpack rectified images
        img_left, img_right = frame.images  # Both images should be rectified

        # Optionally, concatenate images side by side for better spatial context
        # Adjust as needed based on your display preferences
        combined_image = np.hstack((img_left, img_right))

        # Define image dimensions
        height, width, _ = img_left.shape

        for obj in tracked_objects:
            track_id = obj.track_id
            x, y, w, h = obj.bbox
            x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
            label = obj.label
            position_3d = obj.position_3d  # (X, Y, Z) in meters

            # Draw bounding box on the left image (assuming tracked_objects correspond to left image)
            cv2.rectangle(combined_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Prepare label with 3D position
            text = (
                f"ID {track_id} - {label} - "
                f"X: {position_3d[0]:.2f}m, "
                f"Y: {position_3d[1]:.2f}m, "
                f"Z: {position_3d[2]:.2f}m"
            )

            # Calculate position for text to avoid overlapping with bounding box
            text_position = (x1, y1 - 10 if y1 - 10 > 10 else y1 + 10)

            # Draw label background for better readability
            (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(
                combined_image,
                (text_position[0], text_position[1] - text_height - 5),
                (text_position[0] + text_width, text_position[1] + 5),
                (0, 255, 0),
                cv2.FILLED,
            )

            # Put text on the image
            cv2.putText(
                combined_image,
                text,
                text_position,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),  # Black text for contrast
                2,
                cv2.LINE_AA,
            )

        # Display the combined image
        cv2.imshow("Autonomous Perception - Rectified Images", combined_image)

        # Wait for a short period; allow exit with 'q' key
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            self.logger.info("Exit key pressed. Closing visualization window.")
            cv2.destroyAllWindows()
