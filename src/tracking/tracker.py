# tracker.py
import logging

import numpy as np
from scipy.optimize import linear_sum_assignment

from src.data_loader.dataset import TrackedObject
from src.detection.detection_result import DetectionResult


class KalmanFilter3D:
    """Kalman Filter for tracking objects in 3D space with a constant acceleration model."""

    def __init__(self, dt: float = 1.0):
        """
        Initialize the Kalman Filter.

        Args:
            dt (float): Time step between measurements.
        """
        self.dt = dt

        # State vector [x, y, z, vx, vy, vz, ax, ay, az]
        self.x = np.zeros((9, 1))

        # State transition matrix (constant acceleration model)
        dt2 = dt**2 / 2
        self.F = np.array(
            [
                [1, 0, 0, dt, 0, 0, dt2, 0, 0],
                [0, 1, 0, 0, dt, 0, 0, dt2, 0],
                [0, 0, 1, 0, 0, dt, 0, 0, dt2],
                [0, 0, 0, 1, 0, 0, dt, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, dt, 0],
                [0, 0, 0, 0, 0, 1, 0, 0, dt],
                [0, 0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1],
            ]
        )

        # Observation matrix (we can only observe position)
        self.H = np.zeros((3, 9))
        self.H[0, 0] = 1  # x position
        self.H[1, 1] = 1  # y position
        self.H[2, 2] = 1  # z position

        # Process noise covariance
        q = 1e-2  # Process noise scalar
        self.Q = q * np.eye(9)

        # Measurement noise covariance
        r = 1e-1  # Measurement noise scalar
        self.R = r * np.eye(3)

        # Initial estimate error covariance
        self.P = np.eye(9)

    def predict(self):
        """Predict the next state."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z: np.ndarray):
        """
        Update the state with a new measurement.

        Args:
            z (np.ndarray): Measurement vector [x, y, z].
        """
        z = z.reshape((3, 1))
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I = np.eye(self.P.shape[0])
        self.P = (I - K @ self.H) @ self.P

    def get_state(self) -> np.ndarray:
        """Get the current state estimate."""
        return self.x.flatten()

    def get_position(self) -> np.ndarray:
        """Get the predicted 3D position based on the current state."""
        return self.x[:3].flatten()


class Track3D:
    """Represents a single track in 3D space."""

    count = 0  # Class variable to assign unique IDs to each track

    def __init__(self, detection: DetectionResult, dt: float, max_age: int, min_hits: int):
        """
        Initialize a Track3D object.

        Args:
            detection (DetectionResult): The initial detection.
            dt (float): Time step between measurements.
            max_age (int): Maximum frames to keep the track without updates.
            min_hits (int): Minimum number of hits to consider a track as confirmed.
        """
        self.kf = KalmanFilter3D(dt)

        # Initialize state with the detection's 3D position
        self.kf.x[:3] = np.array(detection.position_3d).reshape((3, 1))

        self.time_since_update = 0
        self.id = Track3D.count
        Track3D.count += 1
        self.hits = 1
        self.age = 0
        self.confidence = detection.confidence
        self.class_id = detection.class_id
        self.label = detection.label
        self.bbox = detection.bbox  # Initial bounding box
        self.max_age = max_age
        self.min_hits = min_hits
        self.confirmed = False

    def predict(self):
        """Predict the next state."""
        self.kf.predict()
        self.age += 1
        self.time_since_update += 1

    def update(self, detection: DetectionResult = None):
        """Update the track with a new detection or predict if no detection."""
        if detection is not None:
            self.time_since_update = 0
            self.hits += 1

            # Update the Kalman Filter with the new measurement
            self.kf.update(np.array(detection.position_3d))

            self.confidence = detection.confidence
            self.class_id = detection.class_id
            self.label = detection.label
            self.bbox = detection.bbox  # Update bounding box

            if self.hits >= self.min_hits:
                self.confirmed = True
        else:
            # No detection, just predict
            self.predict()

    def get_state(self) -> np.ndarray:
        """Get the current state estimate."""
        return self.kf.get_state()

    def get_position(self) -> np.ndarray:
        """Get the predicted 3D position based on the current state."""
        return self.kf.x[:3].flatten()


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
        self.max_age = tracking_config.get("max_age", 30)  # Max frames to keep track without update
        self.min_hits = tracking_config.get("min_hits", 3)
        self.dt = tracking_config.get("dt", 1.0)

        self.tracks: list[Track3D] = []

        self.logger.info(
            "3D Object Tracker initialized with Kalman Filters (constant acceleration model).",
        )

    def reset(self):
        """Reset the tracker state."""
        self.tracks = []
        self.logger.info("3D Object Tracker has been reset.")

    def update_tracks(
        self,
        detections: list[DetectionResult],
    ) -> list[TrackedObject]:
        """
        Update tracks with new detections.

        Args:
            detections (List[DetectionResult]): List of current detections.

        Returns:
            List[TrackedObject]: List of active tracked objects.
        """
        # Predict new locations of existing tracks
        for track in self.tracks:
            track.predict()

        # Measurement updates
        matches, unmatched_tracks, unmatched_detections = self.associate_detections_to_tracks(
            detections,
            self.tracks,
        )

        # Update matched tracks with assigned detections
        for track_idx, detection_idx in matches:
            track = self.tracks[track_idx]
            detection = detections[detection_idx]
            track.update(detection)

        # Update unmatched tracks (no detection)
        for track_idx in unmatched_tracks:
            track = self.tracks[track_idx]
            # Continue predicting without update
            # Already predicted above in predict step

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
            position_3d = track.get_position()
            tracked_object = TrackedObject(
                track_id=track.id,
                bbox=track.bbox,
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
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        """
        Assign detections to tracks using the Euclidean distance between predicted and detected positions.

        Args:
            detections (List[DetectionResult]): List of detections.
            tracks (List[Track3D]): List of existing tracks.

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

        cost_matrix = np.zeros((num_tracks, num_detections))

        for t_idx, track in enumerate(tracks):
            track_state = track.get_position()  # Predicted position (x, y, z)
            for d_idx, detection in enumerate(detections):
                detection_position = np.array(detection.position_3d)
                # Compute Euclidean distance
                distance = np.linalg.norm(track_state - detection_position)
                cost_matrix[t_idx, d_idx] = distance

        # Apply gating to limit associations
        max_distance = 5.0  # Maximum allowed distance for association (meters)
        cost_matrix[cost_matrix > max_distance] = max_distance + 1

        # Apply Hungarian algorithm
        row_indices, col_indices = linear_sum_assignment(cost_matrix)

        matches = []
        unmatched_tracks = list(range(num_tracks))
        unmatched_detections = list(range(num_detections))

        for r, c in zip(row_indices, col_indices):
            if cost_matrix[r, c] > max_distance:
                continue
            matches.append((r, c))
            unmatched_tracks.remove(r)
            unmatched_detections.remove(c)

        return matches, unmatched_tracks, unmatched_detections
