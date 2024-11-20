# object_detector.py
import logging

import cv2
import numpy as np

from src.data_loader.dataset import TrackedObject
from src.depth_estimation.depth_estimator import DepthEstimator
from src.detection.detection_result import DetectionResult
from src.models.model import load_model
from src.tracking.tracker import ObjectTracker


class ObjectDetector:
    """Performs object detection, depth estimation, and tracking."""

    SUPPORTED_LABELS = {"person", "car", "bicycle"}

    def __init__(self, config: dict, calib_params: dict[str, np.ndarray]) -> None:
        """
        Initialize the ObjectDetector.

        Args:
            config (dict): Configuration dictionary.
            calib_params (dict[str, np.ndarray]): Stereo calibration parameters.
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

        # Extract focal lengths and principal points from calibration parameters
        self.fx = self.calib_params["P_rect_02"][0, 0]
        self.fy = self.calib_params["P_rect_02"][1, 1]
        self.cx = self.calib_params["P_rect_02"][0, 2]
        self.cy = self.calib_params["P_rect_02"][1, 2]

    def reset(self) -> None:
        """Reset the ObjectDetector state by reinitializing the tracker."""
        self.object_tracker.reset()
        self.logger.info("ObjectTracker has been reset.")

    def detect_and_track(
        self,
        images: tuple[np.ndarray, np.ndarray],
    ) -> tuple[list[DetectionResult], list[TrackedObject], np.ndarray]:
        """
        Perform detection and tracking on the provided stereo images.

        Args:
            images (tuple[np.ndarray, np.ndarray]): A tuple containing left and right rectified images.

        Returns:
            tuple[list[DetectionResult], list[TrackedObject], np.ndarray]: Detections, tracked objects, and depth map.
        """
        img_left, img_right = images

        # Compute disparity map for depth estimation
        disparity_map = self.depth_estimator.compute_disparity_map(img_left, img_right)

        # Compute depth map from disparity map
        depth_map = self.depth_estimator.compute_depth_map(disparity_map)

        # Detect objects in left image
        detections_left = self._detect(img_left)

        # Enhance DetectionResults with accurate depth from depth map
        for det in detections_left:
            bbox = det.bbox
            depth = self._compute_object_depth(bbox, depth_map)
            if depth > 0:
                center_x, center_y = self._get_center(bbox)
                X = (center_x - self.cx) * depth / self.fx
                Y = (center_y - self.cy) * depth / self.fy
                Z = depth
                det.add_3d_position((X, Y, Z))
                self.logger.debug(
                    f"Detection {det.label} at ({center_x:.2f}, {center_y:.2f}) has depth {depth:.2f}m"
                )
            else:
                self.logger.warning(f"Invalid depth for detection {det.label} at bbox {bbox}")
                det.add_3d_position((0.0, 0.0, 0.0))

        # Update tracker with detections
        tracked_objects = self.object_tracker.update_tracks(detections_left)

        return detections_left, tracked_objects, depth_map

    def _detect(self, image: np.ndarray) -> list[DetectionResult]:
        """
        Detect objects in a single image.

        Args:
            image (np.ndarray): The image in which to detect objects.

        Returns:
            list[DetectionResult]: List of detection results.
        """
        try:
            results = self.model(image)
        except Exception as e:
            self.logger.exception(f"Model inference failed: {e}")
            return []

        detections = []

        try:
            # Iterate over the results list
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    cls = int(box.cls.item())
                    label = self.model.names[cls] if cls < len(self.model.names) else "unknown"
                    if label.lower() not in self.SUPPORTED_LABELS:
                        continue
                    confidence = box.conf.item()
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
                        )
                    )
        except AttributeError as e:
            self.logger.exception(f"Unexpected model output structure: {e}")
            return []

        self.logger.debug(f"Detected {len(detections)} objects in image.")

        return detections

    def _get_center(self, bbox: list[float]) -> tuple[float, float]:
        """
        Calculate the center of a bounding box.

        Args:
            bbox (list[float]): Bounding box coordinates [x, y, w, h].

        Returns:
            tuple[float, float]: Center coordinates (x, y).
        """
        x, y, w, h = bbox
        return x + w / 2, y + h / 2

    def _compute_object_depth(self, bbox: list[float], depth_map: np.ndarray) -> float:
        """
        Compute the median depth within the object's bounding box to get a more robust depth estimate.

        Args:
            bbox (list[float]): Bounding box coordinates [x, y, w, h].
            depth_map (np.ndarray): Depth map.

        Returns:
            float: Median depth value within the bounding box.
        """
        x, y, w, h = map(int, bbox)
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(depth_map.shape[1], x + w)
        y2 = min(depth_map.shape[0], y + h)

        depth_roi = depth_map[y1:y2, x1:x2]
        valid_depths = depth_roi[np.isfinite(depth_roi) & (depth_roi > 0)]
        if valid_depths.size > 0:
            median_depth = np.median(valid_depths)
            return median_depth
        else:
            return -1.0  # Invalid depth
