# tracker.py
import logging

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torchvision import models, transforms

from src.data_loader.dataset import TrackedObject
from src.detection.detection_result import DetectionResult


class KalmanFilter3D:
    """Kalman Filter for tracking objects in 3D space with constant velocity model."""

    def __init__(self, dt: float = 1.0):
        """
        Initialize the Kalman Filter.

        Args:
            dt (float): Time step between measurements.

        """
        self.dt = dt
        # State transition matrix (constant velocity model)
        self.F = np.eye(6)
        self.F[0, 3] = dt
        self.F[1, 4] = dt
        self.F[2, 5] = dt

        # Observation matrix
        self.H = np.zeros((3, 6))
        self.H[0, 0] = 1
        self.H[1, 1] = 1
        self.H[2, 2] = 1

        # Process noise covariance
        self.Q = np.eye(6) * 1e-2

        # Measurement noise covariance
        self.R = np.eye(3) * 1e-1

        # Initial state estimate (1D array)
        self.x = np.zeros(6)

        # Initial covariance estimate
        self.P = np.eye(6)

    def predict(self):
        """Predict the next state."""
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q

    def update(self, z: np.ndarray):
        """
        Update the state with a new measurement.

        Args:
            z (np.ndarray): Measurement vector [x, y, z].

        """
        y = z - np.dot(self.H, self.x)
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.x = self.x + np.dot(K, y)
        self.P = self.P - np.dot(np.dot(K, self.H), self.P)

    def get_state(self) -> np.ndarray:
        """Get the current state estimate."""
        return self.x

    def get_covariance(self) -> np.ndarray:
        """Get the current covariance estimate."""
        return self.P


class Track3D:
    """Represents a single track in 3D space."""

    count = 0  # Class variable to assign unique IDs to each track

    def __init__(self, detection: DetectionResult, dt: float, max_age: int, min_hits: int):
        """
        Initialize a Track3D object.

        Args:
            detection (DetectionResult): The initial detection.
            dt (float): Time step between measurements.
            max_age (int): Maximum allowed frames without detections.
            min_hits (int): Minimum number of hits to consider a track as confirmed.

        """
        self.kf = KalmanFilter3D(dt)
        self.kf.x[:3] = np.array(detection.position_3d)  # Corrected assignment
        self.time_since_update = 0
        self.id = Track3D.count
        Track3D.count += 1
        self.history = []
        self.hits = 1
        self.age = 0
        self.confidence = detection.confidence
        self.class_id = detection.class_id
        self.label = detection.label
        self.bbox = detection.bbox
        self.max_age = max_age
        self.min_hits = min_hits
        self.confirmed = False
        self.appearance_feature = detection.appearance_feature  # Store appearance feature

    def predict(self):
        """Predict the next state."""
        self.kf.predict()
        self.age += 1
        self.time_since_update += 1
        self.history.append(self.kf.get_state())

    def update(self, detection: DetectionResult):
        """Update the track with a new detection."""
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.kf.update(np.array(detection.position_3d))
        self.confidence = detection.confidence
        self.class_id = detection.class_id
        self.label = detection.label
        self.bbox = detection.bbox
        if detection.appearance_feature is not None:
            self.appearance_feature = detection.appearance_feature
        if self.hits >= self.min_hits:
            self.confirmed = True

    def get_state(self) -> np.ndarray:
        """Get the current state estimate."""
        return self.kf.get_state()


class ObjectTracker:
    """ObjectTracker class to manage object tracking in 3D space."""

    def __init__(self, config: dict, logger: logging.Logger) -> None:
        """
        Initialize the ObjectTracker.

        Args:
            config (dict): Configuration dictionary for tracking.
            logger (logging.Logger): Logger for logging messages.

        """
        self.logger = logger
        tracking_config = config.get("tracking", {})
        self.max_age = tracking_config.get("max_age", 30)  # Increased to handle occlusions
        self.min_hits = tracking_config.get("min_hits", 3)
        self.dt = tracking_config.get("dt", 1.0)
        self.distance_threshold = tracking_config.get("distance_threshold", 5.0)  # in meters
        self.appearance_weight = tracking_config.get(
            "appearance_weight",
            0.5,
        )  # Weight for appearance cost

        self.tracks: list[Track3D] = []

        # Initialize the appearance feature extractor
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.appearance_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT).to(
            self.device
        )
        self.appearance_model.eval()
        # Define image transforms
        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ],
        )

        self.logger.info(
            "Advanced 3D Object Tracker initialized with Kalman Filters and appearance features.",
        )

    def reset(self):
        """Reset the tracker state."""
        self.tracks = []
        self.logger.info("3D Object Tracker has been reset.")

    def extract_appearance_features(self, detections: list[DetectionResult], frame: np.ndarray):
        """Extract appearance features from detections using a CNN."""
        for detection in detections:
            x, y, w, h = map(int, detection.bbox)
            img_patch = frame[y : y + h, x : x + w]
            if img_patch.size == 0:
                detection.appearance_feature = None
                continue
            img_tensor = self.transform(img_patch).unsqueeze(0).to(self.device)  # type: ignore
            with torch.no_grad():
                feature = self.appearance_model(img_tensor)
            detection.appearance_feature = feature.cpu().numpy().flatten()

    def update_tracks(
        self,
        detections: list[DetectionResult],
        frame: np.ndarray,
    ) -> list[TrackedObject]:
        """
        Update tracks with new detections.

        Args:
            detections (List[DetectionResult]): List of current detections.
            frame (np.ndarray): Current frame image.

        Returns:
            List[TrackedObject]: List of active tracked objects.

        """
        # Extract appearance features
        self.extract_appearance_features(detections, frame)

        # Predict new locations of existing tracks
        for track in self.tracks:
            track.predict()

        # Measurement updates
        matches, unmatched_tracks, unmatched_detections = self.associate_detections_to_tracks(
            detections,
            self.tracks,
            frame,
        )

        # Update matched tracks with assigned detections
        for track_idx, detection_idx in matches:
            track = self.tracks[track_idx]
            detection = detections[detection_idx]
            track.update(detection)

        # Create new tracks for unmatched detections
        for detection_idx in unmatched_detections:
            detection = detections[detection_idx]
            if all(coord == 0.0 for coord in detection.position_3d):
                continue  # Skip detections with invalid 3D positions
            track = Track3D(detection, self.dt, self.max_age, self.min_hits)
            self.tracks.append(track)

        # Remove dead tracks
        self.tracks = [track for track in self.tracks if track.time_since_update <= self.max_age]

        # Prepare output
        tracked_objects = []
        for track in self.tracks:
            if not track.confirmed:
                continue
            state = track.get_state()
            position_3d = (state[0], state[1], state[2])
            tracked_object = TrackedObject(
                track_id=track.id,
                bbox=track.bbox,  # type: ignore
                class_id=track.class_id,
                label=track.label,
                position_3d=position_3d,
            )
            tracked_objects.append(tracked_object)
            self.logger.debug(
                f"Track ID: {track.id}, Label: {track.label}, Position: {position_3d}",
            )

        return tracked_objects

    def associate_detections_to_tracks(
        self,
        detections: list[DetectionResult],
        tracks: list[Track3D],
        frame: np.ndarray,
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        """
        Assigns detections to tracks using combined motion and appearance cost.

        Args:
            detections (List[DetectionResult]): List of detections.
            tracks (List[Track3D]): List of existing tracks.
            frame (np.ndarray): Current frame image.

        Returns:
            Tuple containing:
            - matches (List[Tuple[int, int]]): List of matched track and detection indices.
            - unmatched_tracks (List[int]): List of unmatched track indices.
            - unmatched_detections (List[int]): List of unmatched detection indices.

        """
        if len(tracks) == 0:
            return [], [], list(range(len(detections)))
        if len(detections) == 0:
            return [], list(range(len(tracks))), []

        num_tracks = len(tracks)
        num_detections = len(detections)

        motion_cost = np.zeros((num_tracks, num_detections))
        appearance_cost = np.zeros((num_tracks, num_detections))

        for t_idx, track in enumerate(tracks):
            track_state = track.get_state()[:3]
            track_cov = track.kf.get_covariance()[:3, :3]
            inv_cov = np.linalg.inv(track_cov + np.eye(3) * 1e-6)

            for d_idx, detection in enumerate(detections):
                detection_position = np.array(detection.position_3d)
                # Compute Mahalanobis distance
                diff = detection_position - track_state
                distance = np.sqrt(np.dot(np.dot(diff.T, inv_cov), diff))
                motion_cost[t_idx, d_idx] = distance

                # Compute appearance cost if features are available
                if (
                    track.appearance_feature is not None
                    and detection.appearance_feature is not None
                ):
                    # Use cosine similarity
                    track_feat = track.appearance_feature.reshape(1, -1)
                    det_feat = detection.appearance_feature.reshape(1, -1)
                    appearance_distance = (
                        1
                        - F.cosine_similarity(
                            torch.from_numpy(track_feat),
                            torch.from_numpy(det_feat),
                            dim=1,
                        ).item()
                    )
                else:
                    # Assign maximum cost if features are missing
                    appearance_distance = 1.0  # Maximum cosine distance

                appearance_cost[t_idx, d_idx] = appearance_distance

        # Normalize costs
        if np.max(motion_cost) > 0:
            motion_cost /= np.max(motion_cost)
        if np.max(appearance_cost) > 0:
            appearance_cost /= np.max(appearance_cost)

        # Compute combined cost
        total_cost = (
            1 - self.appearance_weight
        ) * motion_cost + self.appearance_weight * appearance_cost

        # Apply gating to limit associations
        max_cost = 1.0  # Since costs are normalized
        total_cost[total_cost > max_cost] = max_cost + 1

        # Apply Hungarian algorithm
        row_indices, col_indices = linear_sum_assignment(total_cost)

        matches = []
        unmatched_tracks = list(range(num_tracks))
        unmatched_detections = list(range(num_detections))

        for r, c in zip(row_indices, col_indices, strict=False):
            if total_cost[r, c] > max_cost:
                continue
            matches.append((r, c))
            unmatched_tracks.remove(r)
            unmatched_detections.remove(c)

        return matches, unmatched_tracks, unmatched_detections
