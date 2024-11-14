import logging

import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLO

from src.models.yolov8 import model as yolov8


class DetectionResult:
    def __init__(self, bbox: list[float], confidence: float, class_id: int):
        self.bbox = bbox  # [x1, y1, x2, y2]
        self.confidence = confidence
        self.class_id = class_id


class TrackedObject:
    def __init__(self, track_id: int, bbox: list[int], class_id: int, label: str):
        self.track_id = track_id
        self.bbox = bbox
        self.class_id = class_id
        self.label = label


class ObjectDetector:
    SUPPORTED_LABELS = {"person", "car", "bicycle"}

    def __init__(self, config: dict):
        self.model_name = config.get("model", "yolov8")
        self.conf_threshold = config.get("confidence_threshold", 0.5)
        self.nms_threshold = config.get("nms_threshold", 0.45)
        self.logger = logging.getLogger("autonomous_perception.detection")
        self.model = self._load_model()
        self.tracker = self._initialize_tracker()

    def _load_model(self) -> YOLO:
        self.logger.info(f"Loading detection model: {self.model_name}")
        if self.model_name.lower() == "yolov8":
            model = yolov8.get_model()
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")
        self.logger.info("Detection model loaded successfully")
        return model

    def _initialize_tracker(self) -> DeepSort:
        self.logger.info("Initializing DeepSort tracker")
        tracker = DeepSort(max_age=30, n_init=3, nn_budget=100)
        self.logger.info("DeepSort tracker initialized successfully")
        return tracker

    def detect_and_track(
        self,
        images: tuple[np.ndarray, np.ndarray],
    ) -> tuple[list[DetectionResult], list[TrackedObject]]:
        """
        Perform detection and tracking on the provided images.

        Args:
            images (Tuple[np.ndarray, np.ndarray]): A tuple containing left and right images.

        Returns:
            Tuple[List[DetectionResult], List[TrackedObject]]: A tuple with detections and tracked objects.

        """
        # Perform detection on the left rectified image
        img_left, _ = images
        self.logger.debug("Performing detection on left image")
        results = self.model(img_left)

        detections = self._process_detections(results)
        tracked_objects = self._update_tracks(detections, img_left)

        return detections, tracked_objects

    def _process_detections(self, results) -> list[DetectionResult]:
        """
        Process YOLO detection results into DetectionResult objects.

        Args:
            results: YOLO detection results.

        Returns:
            List[DetectionResult]: A list of DetectionResult instances.

        """
        detections = []
        if not isinstance(results, list):
            self.logger.error(f"Expected results to be a list, but got {type(results)}")
            raise TypeError(f"Expected results to be a list, but got {type(results)}")

        if len(results) == 0:
            self.logger.warning("No results returned from the model.")
            return detections

        first_result = results[0]
        if not hasattr(first_result, "boxes"):
            self.logger.error("The first result does not have a 'boxes' attribute.")
            raise AttributeError("The first result does not have a 'boxes' attribute.")

        for box in first_result.boxes:
            cls = int(box.cls[0])
            label = self.model.names[cls] if cls < len(self.model.names) else "unknown"
            if label in self.SUPPORTED_LABELS:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                confidence = box.conf[0].item()
                if confidence >= self.conf_threshold:
                    detections.append(
                        DetectionResult(bbox=[x1, y1, x2, y2], confidence=confidence, class_id=cls),
                    )
        self.logger.debug(f"Detections found: {len(detections)}")
        return detections

    def _update_tracks(
        self,
        detections: list[DetectionResult],
        frame: np.ndarray,
    ) -> list[TrackedObject]:
        """
        Update DeepSort tracker with current detections and return tracked objects.

        Args:
            detections (List[DetectionResult]): Current frame detections.
            frame (np.ndarray): The current image frame.

        Returns:
            List[TrackedObject]: A list of TrackedObject instances.

        """
        # Prepare detections for DeepSort
        prepared_detections = []
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            bbox = [x1, y1, x2 - x1, y2 - y1]  # Convert to [x, y, w, h]
            prepared_detections.append((bbox, det.confidence, det.class_id))

        # Update tracker
        tracks = self.tracker.update_tracks(prepared_detections, frame=frame)
        tracked_objects = []

        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            bbox = track.to_ltwh()  # [x, y, w, h]
            class_id = track.get_det_class()
            label = (
                self.model.names[class_id]
                if class_id is not None and class_id < len(self.model.names)
                else "unknown"
            )
            tracked_objects.append(
                TrackedObject(track_id=track_id, bbox=bbox, class_id=class_id, label=label),
            )

        self.logger.debug(f"Tracked objects count: {len(tracked_objects)}")
        return tracked_objects
