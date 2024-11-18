import logging

import cv2
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort

from src.data_loader.dataset import TrackedObject
from src.models.model import load_model  # Ensure correct import path


class DetectionResult:
    """DetectionResult class to store detection information."""

    def __init__(self, bbox: list[float], confidence: float, class_id: int) -> None:
        """
        Initialize a DetectionResult object.

        Args:
            bbox (List[float]): Bounding box coordinates [x1, y1, x2, y2].
            confidence (float): Confidence score of the detection.
            class_id (int): Class ID of the detected object.

        """
        self.bbox = bbox  # [x1, y1, x2, y2]
        self.confidence = confidence
        self.class_id = class_id
        self.position_3d: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def add_3d_position(self, position_3d: tuple[float, float, float]) -> None:
        """
        Add 3D position to the detection result.

        Args:
            position_3d (Tuple[float, float, float]): 3D coordinates (X, Y, Z).

        """
        self.position_3d = position_3d


class ObjectDetector:
    """ObjectDetector class to perform object detection and tracking."""

    SUPPORTED_LABELS = {"person", "car", "bicycle"}

    def __init__(self, config: dict, calib_params: dict[str, np.ndarray]) -> None:
        """
        Initialize the ObjectDetector with detection and tracking models.

        Args:
            config (Dict): Configuration dictionary.
            calib_params (Dict[str, np.ndarray]): Stereo calibration parameters.

        """
        self.conf_threshold = config["detection"].get("confidence_threshold", 0.5)
        self.nms_threshold = config["detection"].get("nms_threshold", 0.4)
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
        self.model = load_model(model_path)
        self.model.conf = self.conf_threshold  # Set confidence threshold
        self.model.iou = self.nms_threshold  # Set NMS threshold
        self.logger.info(f"YOLO model loaded from {model_path}")

        # Initialize DeepSort tracker
        tracking_config = config.get("tracking", {})
        self.tracker = DeepSort(
            max_age=tracking_config.get(
                "max_age",
                60,
            ),  # Increased max_age for better occlusion handling
            n_init=tracking_config.get("n_init", 5),  # Increased n_init to confirm tracks
            nn_budget=tracking_config.get(
                "nn_budget",
                200,
            ),  # Increased nn_budget for better appearance matching
        )
        self.logger.info("DeepSort tracker initialized")

        # Calibration parameters
        self.calib_params = calib_params

        # Compute focal length and baseline from rectified projection matrices
        try:
            P_rect_02 = calib_params["P_rect_02"]  # 3x4 matrix
            P_rect_03 = calib_params["P_rect_03"]  # 3x4 matrix
            self.focal_length = P_rect_02[0, 0]  # fx from rectified left projection matrix
            Tx_left = P_rect_02[0, 3]
            Tx_right = P_rect_03[0, 3]
            self.baseline = np.abs(Tx_right - Tx_left) / self.focal_length
            self.logger.info(f"Focal length set to {self.focal_length}")
            self.logger.info(f"Baseline set to {self.baseline} meters")
        except KeyError:
            self.logger.exception("Error computing focal length and baseline.")
            raise
        except Exception:
            self.logger.exception("Error initializing ObjectDetector.")
            raise

        # Initialize StereoSGBM matcher with improved parameters
        self.stereo_matcher = cv2.StereoSGBM.create(
            minDisparity=0,
            numDisparities=128,  # Increased for better depth resolution
            blockSize=7,  # Increased block size for smoother disparity
            P1=8 * 3 * 7**2,
            P2=32 * 3 * 7**2,
            disp12MaxDiff=1,
            uniquenessRatio=15,  # Increased to reduce false matches
            speckleWindowSize=100,
            speckleRange=32,
            preFilterCap=63,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
        )
        self.logger.info("StereoSGBM matcher initialized with improved parameters")

        # Initialize WLS Filter for disparity map refinement
        self.wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=self.stereo_matcher)
        self.wls_filter.setLambda(8000)  # Increased lambda for smoother disparity
        self.wls_filter.setSigmaColor(1.5)  # Adjusted sigmaColor for edge-preserving

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
        disparity_map = self.compute_disparity_map(img_left, img_right)

        # Detect objects in left image
        detections_left = self._detect(img_left)

        # Enhance DetectionResults with accurate depth from disparity map
        for det in detections_left:
            center_x, center_y = self._get_center(det.bbox)
            if 0 <= center_y < disparity_map.shape[0] and 0 <= center_x < disparity_map.shape[1]:
                disparity = disparity_map[int(center_y), int(center_x)]
                if disparity > 0:
                    depth = self.focal_length * self.baseline / disparity
                    X = (
                        (center_x - self.calib_params["P_rect_02"][0, 2])
                        * depth
                        / self.focal_length
                    )
                    Y = (
                        (center_y - self.calib_params["P_rect_02"][1, 2])
                        * depth
                        / self.calib_params["P_rect_02"][1, 1]
                    )
                    det.add_3d_position((X, Y, depth))
                    self.logger.debug(
                        f"Detection ID {det.class_id} at ({center_x}, {center_y}) has depth {depth:.2f}m",
                    )
                else:
                    self.logger.warning(
                        f"Invalid disparity for detection at ({center_x}, {center_y})",
                    )
                    det.add_3d_position((0.0, 0.0, 0.0))
            else:
                self.logger.warning(
                    f"Detection center ({center_x}, {center_y}) is out of disparity map bounds.",
                )
                det.add_3d_position((0.0, 0.0, 0.0))

        # Update tracker with detections
        tracked_objects = self._update_tracks(detections_left, img_left)

        return detections_left, tracked_objects, disparity_map

    def _detect(self, image: np.ndarray) -> list[DetectionResult]:
        """
        Detect objects in a single image.

        Args:
            image (np.ndarray): The image in which to detect objects.

        Returns:
            List[DetectionResult]: List of detection results.

        """
        results = self.model(image)
        detections = []

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
                detections.append(
                    DetectionResult(bbox=[x1, y1, x2, y2], confidence=confidence, class_id=cls),
                )

        self.logger.debug(f"Detected {len(detections)} objects in image.")
        return detections

    def compute_disparity_map(self, img_left: np.ndarray, img_right: np.ndarray) -> np.ndarray:
        """
        Compute the disparity map from rectified stereo images.

        Args:
            img_left (np.ndarray): Rectified left image.
            img_right (np.ndarray): Rectified right image.

        Returns:
            np.ndarray: Computed disparity map.

        """
        # Convert images to grayscale
        gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

        # Compute disparity using StereoSGBM
        disparity_left = (
            self.stereo_matcher.compute(gray_left, gray_right).astype(np.float32) / 16.0
        )
        self.logger.debug("Left disparity map computed.")

        # Compute disparity for right image
        stereo_matcher_right = cv2.ximgproc.createRightMatcher(self.stereo_matcher)
        disparity_right = (
            stereo_matcher_right.compute(gray_right, gray_left).astype(np.float32) / 16.0
        )
        self.logger.debug("Right disparity map computed.")

        # Apply WLS filter to refine disparity map
        disparity_filtered = self.wls_filter.filter(
            disparity_left,
            img_left,
            disparity_map_right=disparity_right,
        )
        self.logger.debug("Disparity map filtered using WLS filter.")

        # Replace negative disparities with zero
        disparity_filtered[disparity_filtered < 0] = 0
        return disparity_filtered

    def _get_center(self, bbox: list[float]) -> tuple[float, float]:
        """
        Calculate the center of a bounding box.

        Args:
            bbox (List[float]): Bounding box coordinates [x1, y1, x2, y2].

        Returns:
            Tuple[float, float]: Center coordinates (x, y).

        """
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def _update_tracks(
        self,
        detections: list[DetectionResult],
        frame: np.ndarray,
    ) -> list[TrackedObject]:
        """
        Update DeepSort tracker with current detections and return tracked objects with 3D positions.

        Args:
            detections (List[DetectionResult]): Current frame detections with depth.
            frame (np.ndarray): The current image frame (left image).

        Returns:
            List[TrackedObject]: List of tracked objects with 3D positions.

        """
        detections_for_tracker = []
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            bbox = [x1, y1, x2 - x1, y2 - y1]  # [left, top, width, height]
            confidence = det.confidence
            class_id = det.class_id
            detections_for_tracker.append((bbox, confidence, class_id))

        self.logger.debug(f"detections_for_tracker: {detections_for_tracker}")

        # Update tracker with detections
        try:
            tracks = self.tracker.update_tracks(detections_for_tracker, frame=frame)
        except Exception:
            self.logger.exception("Failed to update tracks with current detections.")
            tracks = []

        tracked_objects = []
        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            bbox = track.to_ltwh()  # [left, top, width, height]
            class_id = track.get_det_class()
            label = (
                self.model.names[class_id] if 0 <= class_id < len(self.model.names) else "unknown"
            )

            # Retrieve 3D position from the latest detection
            # Assuming that detections are ordered, and the last matched detection has the
            # most recent position
            matched_detection = None
            for det in detections:
                det_center = self._get_center(det.bbox)
                track_center = (bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2)
                distance = np.linalg.norm(np.array(det_center) - np.array(track_center))
                if distance < float("inf"):  # Threshold for matching (pixels)
                    matched_detection = det
                    break
            position_3d = matched_detection.position_3d if matched_detection else (0.0, 0.0, 0.0)

            tracked_object = TrackedObject(
                track_id=track_id,
                bbox=[int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])],
                class_id=class_id,
                label=label,
                position_3d=position_3d,
            )
            tracked_objects.append(tracked_object)
            self.logger.debug(
                f"Tracked Object ID: {track_id}, Label: {label}, Position: {position_3d}",
            )

        self.logger.info(f"Total tracked objects: {len(tracked_objects)}")
        return tracked_objects
