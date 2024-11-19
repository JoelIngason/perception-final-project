# object_detector.py
import logging

import numpy as np

from src.data_loader.dataset import TrackedObject
from src.depth_estimation.depth_estimator import DepthEstimator
from src.detection.detection_result import DetectionResult
from src.models.model import load_model  # Ensure correct import path
from src.tracking.tracker import ObjectTracker


class ObjectDetector:
    """ObjectDetector class to perform object detection and integrate depth estimation and tracking."""

    SUPPORTED_LABELS = {"person", "car", "bicycle"}

    def __init__(self, config: dict, calib_params: dict[str, np.ndarray]) -> None:
        """
        Initialize the ObjectDetector with detection, depth estimation, and tracking models.

        Args:
            config (Dict): Configuration dictionary.
            calib_params (Dict[str, np.ndarray]): Stereo calibration parameters.

        """
        self.conf_threshold = config["detection"].get("confidence_threshold", 0.5)
        self.nms_threshold = config["detection"].get("nms_threshold", 0.4)

        # Setup logger
        self.logger = logging.getLogger("autonomous_perception.detection")
        self.logger.setLevel(logging.INFO)
        if not self.logger.hasHandlers():
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        # Load YOLO model
        model_path = config["detection"]["classification"].get(
            "model_path",
            "src/models/yolov8/yolov8n.pt",
        )
        try:
            self.model = load_model(model_path)
            self.model.conf = self.conf_threshold  # Set confidence threshold
            self.model.iou = self.nms_threshold  # Set NMS threshold
            self.logger.info(f"YOLO model loaded from {model_path}")
        except Exception as e:
            self.logger.exception(f"Failed to load YOLO model from {model_path}: {e}")
            raise

        # Initialize DepthEstimator
        self.depth_estimator = DepthEstimator(calib_params, self.logger)

        # Initialize ObjectTracker
        self.object_tracker = ObjectTracker(config, self.logger)

        # Calibration parameters
        self.calib_params = calib_params

    def reset(self) -> None:
        """Reset the ObjectDetector state by reinitializing the tracker."""
        self.object_tracker = ObjectTracker(config={}, logger=self.logger)
        self.logger.info("ObjectTracker has been reset.")

    def detect_and_track(
        self,
        images: tuple[np.ndarray, np.ndarray],
    ) -> tuple[list[DetectionResult], list[TrackedObject], np.ndarray]:
        """
        Perform detection and tracking on the provided stereo images.

        Args:
            images (Tuple[np.ndarray, np.ndarray]): A tuple containing left and right rectified images.

        Returns:
            Tuple[List[DetectionResult], List[TrackedObject], np.ndarray]: Detections, tracked objects, and disparity map.

        """
        img_left, img_right = images

        # Compute disparity map for depth estimation
        disparity_map = self.depth_estimator.compute_disparity_map(img_left, img_right)

        # Detect objects in left image
        detections_left = self._detect(img_left)

        # Enhance DetectionResults with accurate depth from disparity map
        for det in detections_left:
            center_x, center_y = self._get_center(det.bbox)
            if 0 <= center_y < disparity_map.shape[0] and 0 <= center_x < disparity_map.shape[1]:
                disparity = disparity_map[int(center_y), int(center_x)]
                if disparity > 0:
                    depth = self.depth_estimator.compute_depth(disparity)
                    X = (
                        (center_x - self.calib_params["P_rect_02"][0, 2])
                        * depth
                        / self.depth_estimator.focal_length
                    )
                    Y = (
                        (center_y - self.calib_params["P_rect_02"][1, 2])
                        * depth
                        / self.calib_params["P_rect_02"][1, 1]
                    )
                    det.add_3d_position((X, Y, depth))
                    self.logger.debug(
                        f"Detection {det.label} at ({center_x}, {center_y}) has depth {depth:.2f}m",
                    )
                else:
                    self.logger.warning(
                        f"Invalid disparity for detection at ({center_x}, {center_y}) : {disparity}",
                    )
                    det.add_3d_position((0.0, 0.0, 0.0))
            else:
                self.logger.warning(
                    f"Detection center ({center_x}, {center_y}) is out of disparity map bounds.",
                )
                det.add_3d_position((0.0, 0.0, 0.0))

        # Update tracker with detections
        tracked_objects = self.object_tracker.update_tracks(detections_left, img_left)

        return detections_left, tracked_objects, disparity_map

    def _detect(self, image: np.ndarray) -> list[DetectionResult]:
        """
        Detect objects in a single image.

        Args:
            image (np.ndarray): The image in which to detect objects.

        Returns:
            List[DetectionResult]: List of detection results.

        """
        try:
            results = self.model(image)
        except Exception as e:
            self.logger.exception(f"Model inference failed: {e}")
            return []

        detections = []

        try:
            for result in results:
                for box in result.boxes:
                    cls = int(box.cls[0])
                    label = self.model.names[cls] if cls < len(self.model.names) else "unknown"
                    if label.lower() not in self.SUPPORTED_LABELS:
                        continue
                    confidence = box.conf[0].item()
                    if confidence < self.conf_threshold:
                        continue
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    x, y, w, h = x1, y1, x2 - x1, y2 - y1
                    detections.append(
                        DetectionResult(
                            bbox=[x, y, w, h],
                            confidence=confidence,
                            class_id=cls,
                            label=label,
                        ),
                    )
        except AttributeError as e:
            self.logger.exception(f"Unexpected model output structure: {e}")
            return []

        self.logger.debug(f"Detected {len(detections)} objects in image.")
        return detections

    def _get_center(self, bbox: list[float]) -> tuple[int, int]:
        """
        Calculate the center of a bounding box.

        Args:
            bbox (List[float]): Bounding box coordinates [x, y, w, h].

        Returns:
            Tuple[int, int]: Center coordinates (x, y).

        """
        x, y, w, h = bbox
        return int(x + w / 2), int(y + h / 2)
