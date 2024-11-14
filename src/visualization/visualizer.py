import logging

import cv2

# Import Frame and TrackedObject classes
from src.data_loader.dataset import Frame
from src.detection.object_detector import TrackedObject


class Visualizer:
    def __init__(self):
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
        Display the frame with tracked objects.

        Args:
            frame (Frame): Frame object containing images and metadata.
            tracked_objects (List[TrackedObject]): List of tracked objects from DeepSort.

        """
        img_left, _ = frame.images  # Assuming we visualize the left image

        for obj in tracked_objects:
            track_id = obj.track_id
            x, y, w, h = obj.bbox
            x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
            label = obj.label

            # Draw bounding box
            cv2.rectangle(img_left, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw label
            text = f"ID {track_id} - {label}"
            cv2.putText(
                img_left,
                text,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                2,
            )

        # Display the image
        cv2.imshow("Autonomous Perception", img_left)

        # Wait for a short period; allow exit with 'q' key
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            self.logger.info("Exit key pressed. Closing visualization window.")
            cv2.destroyAllWindows()
