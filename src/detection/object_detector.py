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
        self.position_3d = None

    def add_3d_position(self, position_3d: tuple[float, float, float]):
        self.position_3d = position_3d


class TrackedObject:
    def __init__(
        self,
        track_id: int,
        bbox: list[int],
        class_id: int,
        label: str,
        position_3d: tuple[float, float, float],
    ):
        """
        Initialize a TrackedObject with 2D bounding box and 3D position.

        Args:
            track_id (int): Unique identifier for the track.
            bbox (list[int]): Bounding box in [x, y, w, h] format.
            class_id (int): Class ID of the detected object.
            label (str): Label/name of the detected object.
            position_3d (tuple[float, float, float]): 3D position (X, Y, Z) in meters.

        """
        self.track_id = track_id
        self.bbox = bbox  # [x, y, w, h]
        self.class_id = class_id
        self.label = label
        self.position_3d = position_3d  # (X, Y, Z)


class ObjectDetector:
    SUPPORTED_LABELS = {"person", "car", "bicycle"}

    def __init__(self, config: dict, calib_params: dict[str, np.ndarray], sequence_number: int):
        """
        Initialize the ObjectDetector with detection and tracking models.

        Args:
            config (dict): Configuration dictionary.
            calib_params (dict[str, np.ndarray]): Stereo calibration parameters.

        """
        self.model_name = config.get("model", "yolov8")
        self.conf_threshold = config.get("confidence_threshold", 0.5)
        self.nms_threshold = config.get("nms_threshold", 0.45)
        self.logger = logging.getLogger("autonomous_perception.detection")
        self.model = self._load_model()
        self.tracker = self._initialize_tracker()
        self.calib_params = calib_params
        self.focal_length = calib_params[f"K_{sequence_number}"][
            0,
            0,
        ]  # Focal length from camera matrix
        self.baseline = np.linalg.norm(calib_params[f"T_{sequence_number}"])

    def _load_model(self) -> YOLO:
        """
        Load the YOLO model.

        Returns:
            YOLO: Loaded YOLO model.

        """
        self.logger.info(f"Loading detection model: {self.model_name}")
        if self.model_name.lower() == "yolov8":
            model = yolov8.get_model()
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")
        self.logger.info("Detection model loaded successfully")
        return model

    def _initialize_tracker(self) -> DeepSort:
        """
        Initialize the DeepSort tracker.

        Returns:
            DeepSort: Initialized DeepSort tracker.

        """
        self.logger.info("Initializing DeepSort tracker")
        tracker = DeepSort(max_age=30, n_init=3, nn_budget=100)
        self.logger.info("DeepSort tracker initialized successfully")
        return tracker

    def detect_and_track(
        self,
        images: tuple[np.ndarray, np.ndarray],
    ) -> tuple[list[DetectionResult], list[TrackedObject]]:
        """
        Perform detection and tracking on the provided stereo images.

        Args:
            images (Tuple[np.ndarray, np.ndarray]): A tuple containing left and right rectified images.

        Returns:
            Tuple[List[DetectionResult], List[TrackedObject]]: A tuple with detections and tracked objects with 3D positions.

        """
        # Perform detection on both left and right rectified images
        img_left, img_right = images
        self.logger.debug("Performing detection on left image")
        results_left = self.model(img_left)
        self.logger.debug("Performing detection on right image")
        results_right = self.model(img_right)

        detections_left = self._process_detections(results_left)
        detections_right = self._process_detections(results_right)

        # Match detections between left and right images
        matched_detections = self._match_detections(detections_left, detections_right)

        # Update tracker with matched detections and compute 3D positions
        tracked_objects = self._update_tracks(matched_detections, img_left)

        return matched_detections, tracked_objects

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

    def _match_detections(
        self,
        detections_left: list[DetectionResult],
        detections_right: list[DetectionResult],
    ) -> list[DetectionResult]:
        """
        Match detections between left and right images based on class and proximity.

        Args:
            detections_left (List[DetectionResult]): Detections from the left image.
            detections_right (List[DetectionResult]): Detections from the right image.

        Returns:
            List[DetectionResult]: Matched detections with depth information.

        """
        matched_detections = []
        for det_left in detections_left:
            # Compute center of the bounding box in left image
            center_left = (
                (det_left.bbox[0] + det_left.bbox[2]) / 2,
                (det_left.bbox[1] + det_left.bbox[3]) / 2,
            )

            # Find the best matching detection in the right image
            min_disparity = float("inf")
            best_match = None
            for det_right in detections_right:
                if det_right.class_id != det_left.class_id:
                    continue  # Only match same class
                center_right = (
                    (det_right.bbox[0] + det_right.bbox[2]) / 2,
                    (det_right.bbox[1] + det_right.bbox[3]) / 2,
                )
                # Compute horizontal disparity
                disparity = center_left[0] - center_right[0]
                if 0 < disparity < min_disparity:
                    min_disparity = disparity
                    best_match = det_right

            if best_match and min_disparity > 0:
                # Compute depth (Z) using disparity
                Z = (self.focal_length * self.baseline) / min_disparity
                # Compute real-world X and Y (assuming Z is depth)
                # Assuming camera coordinates where X and Y are proportional to pixel coordinates
                X = (center_left[0] - self.calib_params["K_02"][0, 2]) * Z / self.focal_length
                Y = (
                    (center_left[1] - self.calib_params["K_02"][1, 2])
                    * Z
                    / self.calib_params["K_02"][1, 1]
                )
                position_3d = (X, Y, Z)
                # Create a new DetectionResult with depth information
                matched_det = DetectionResult(
                    bbox=det_left.bbox,
                    confidence=det_left.confidence,
                    class_id=det_left.class_id,
                )
                matched_det.add_3d_position(position_3d)
                matched_detections.append(matched_det)
                self.logger.debug(f"Matched detection: {matched_det.bbox} with depth {Z:.2f}m")
            else:
                self.logger.debug(f"No match found for detection: {det_left.bbox}")
        self.logger.debug(f"Total matched detections: {len(matched_detections)}")
        return matched_detections

    def _update_tracks(
        self,
        matched_detections: list[DetectionResult],
        frame: np.ndarray,
    ) -> list[TrackedObject]:
        """
        Update DeepSort tracker with current matched detections and return tracked objects with 3D positions.

        Args:
            matched_detections (List[DetectionResult]): Current frame matched detections with depth.
            frame (np.ndarray): The current image frame (left image).

        Returns:
            List[TrackedObject]: A list of TrackedObject instances with 3D positions.

        """
        # Prepare detections for DeepSort
        prepared_detections = []
        for det in matched_detections:
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
            # Retrieve the corresponding depth information
            # Assuming that matched_detections are in the same order as tracks
            # Alternatively, match based on track's bbox with matched_detections
            # Here, we'll find the closest detection
            track_center = (
                (bbox[0] + bbox[2]) / 2,
                (bbox[1] + bbox[3]) / 2,
            )
            min_distance = float("inf")
            position_3d = (0.0, 0.0, 0.0)
            for det in matched_detections:
                det_center = (
                    (det.bbox[0] + det.bbox[2]) / 2,
                    (det.bbox[1] + det.bbox[3]) / 2,
                )
                distance = np.linalg.norm(np.array(track_center) - np.array(det_center))
                if distance < min_distance:
                    min_distance = distance
                    position_3d = getattr(det, "position_3d", (0.0, 0.0, 0.0))

            tracked_objects.append(
                TrackedObject(
                    track_id=track_id,
                    bbox=bbox,
                    class_id=class_id,
                    label=label,
                    position_3d=position_3d,
                ),
            )

        self.logger.debug(f"Tracked objects count: {len(tracked_objects)}")
        return tracked_objects
