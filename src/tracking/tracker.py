import logging

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from src.data_loader.dataset import TrackedObject
from src.detection.detection_result import DetectionResult
from src.tracking.kalman_filter import KalmanFilter3D


class Track3D:
    """Represents a single track in 3D space."""

    count = 0  # Class variable to assign unique IDs to each track

    def __init__(
        self,
        detection: DetectionResult,
        dt: float,
        max_age: int,
        min_hits: int,
        projection_matrix: np.ndarray,
    ) -> None:
        """
        Initialize a Track3D object.

        Args:
            detection (DetectionResult): The initial detection.
            dt (float): Time step between measurements.
            max_age (int): Maximum frames to keep the track without updates.
            min_hits (int): Minimum number of hits to consider a track as confirmed.
            projection_matrix (np.ndarray): Camera projection matrix (3x4).

        """
        self.logger = logging.getLogger("autonomous_perception.tracker.track3d")
        self.kf = KalmanFilter3D(dt)
        # Initialize state with the detection's 3D position
        self.kf.kf.statePost[:3] = np.array(detection.position_3d).reshape((3, 1))
        self.time_since_update = 0
        self.id = Track3D.count
        Track3D.count += 1
        self.hits = 1
        self.age = 0
        self.confidence = detection.confidence
        self.class_id = detection.class_id
        self.label = detection.label
        self.bbox = detection.bbox  # Initial bounding box
        self.appearance_feature = detection.appearance_feature
        self.dimensions = detection.dimensions
        self.mask = detection.mask
        self.max_age = max_age
        self.min_hits = min_hits
        self.confirmed = False
        self.projection_matrix = projection_matrix  # Store the projection matrix

    def predict(self):
        """Predict the next state."""
        self.kf.predict()
        self.age += 1
        self.time_since_update += 1

        # Check if the predicted position is within image bounds
        if not self._is_within_image_bounds(self.bbox):
            self.logger.info(
                f"Track ID {self.id} predicted outside image bounds: {self.bbox}.",
            )
            self.time_since_update = self.max_age + 1  # Mark for deletion

    def _is_within_image_bounds(self, bbox: list[int]) -> bool:
        """Check if the predicted position is within image bounds."""
        x1, y1, x2, y2 = bbox
        x = (x1 + x2) / 2
        y = (y1 + y2) / 2

        image_width = 1224
        image_height = 370
        return not (x < 0 or x > image_width or y < 0 or y > image_height)

    def update(self, detection: DetectionResult | None = None) -> None:
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
            self.appearance_feature = detection.appearance_feature
            self.dimensions = detection.dimensions
            self.mask = detection.mask
            if self.hits >= self.min_hits:
                self.confirmed = True
        else:
            # No detection, just predict
            old_position = self.get_position()
            self.predict()
            # TODO: predict bounding box, for now just move all the coordinates by the
            # same amount as the position
            new_position = self.get_position()
            delta = new_position - old_position
            self.bbox = [coord + delta[i] for i, coord in enumerate(self.bbox)]

    def get_position(self) -> np.ndarray:
        """Get the predicted 3D position based on the current state."""
        return self.kf.get_position()

    def get_appearance_feature(self) -> np.ndarray:
        """Get the appearance feature of the track."""
        return (
            self.appearance_feature
            if self.appearance_feature is not None and self.appearance_feature.size > 0
            else np.array([])
        )


class ObjectTracker:
    """ObjectTracker class to manage object tracking in 3D space."""

    def __init__(self, config: dict, calib_params: dict[str, np.ndarray], camera: str) -> None:
        """
        Initialize the ObjectTracker.

        Args:
            config (dict): Configuration dictionary for tracking.
            calib_params (Dict[str, np.ndarray]): Stereo calibration parameters.
            camera (str): 'left' or 'right' camera.

        """
        self.logger = logging.getLogger("autonomous_perception.tracker")
        tracking_config = config.get("tracking", {})
        self.max_age = tracking_config.get("max_age", 60)
        self.min_hits = tracking_config.get("min_hits", 1)  # Lowered from 3 to 1 for debugging
        self.dt = tracking_config.get("dt", 1.0)
        self.tracks: list[Track3D] = []
        self.logger.info(
            "3D Object Tracker initialized with Kalman Filters (constant velocity model).",
        )

        # Retrieve the projection matrix from config
        if camera == "left":
            self.projection_matrix = calib_params.get("P_rect_02")
        elif camera == "right":
            self.projection_matrix = calib_params.get("P_rect_03")
        else:
            CAMERA_ERROR = "Invalid camera selection"
            raise ValueError(CAMERA_ERROR)

        if self.projection_matrix is None:
            self.projection_matrix = np.eye(3, 4, dtype=np.float32)  # Default to identity matrix

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
            detections (list[DetectionResult]): list of current detections.

        Returns:
            list[TrackedObject]: list of active tracked objects.

        """
        # Predict new locations of existing tracks
        for track in self.tracks:
            track.predict()

        self.logger.debug(f"Predicted {len(self.tracks)} tracks.")

        # Associate detections to tracks
        matched, unmatched_tracks, unmatched_detections = self.associate_detections_to_tracks(
            detections,
            self.tracks,
        )
        combined_len = len(matched) + len(unmatched_tracks) + len(unmatched_detections)
        assert combined_len >= len(
            self.tracks
        ), "Some tracks were not considered in the association."

        self.logger.debug(f"Number of matches: {len(matched)}")
        self.logger.debug(f"Number of unmatched tracks: {len(unmatched_tracks)}")
        self.logger.debug(f"Number of unmatched detections: {len(unmatched_detections)}")

        # Update matched tracks with assigned detections
        for track_idx, det_idx in matched:
            track = self.tracks[track_idx]
            detection = detections[det_idx]
            track.update(detection)
            self.logger.debug(f"Updated Track ID {track.id} with Detection ID {detection}")

        # Handle unmatched detections by creating new tracks
        for det_idx in unmatched_detections:
            detection = detections[det_idx]
            if detection.position_3d is None or all(
                coord == 0.0 for coord in detection.position_3d
            ):
                self.logger.debug(
                    f"Skipping Detection {detection.class_id} with invalid position_3d.",
                )
                continue  # Skip detections with invalid 3D positions
            if self.projection_matrix is None:
                self.logger.warning(
                    "Projection matrix is missing. Cannot create new tracks without 3D calibration.",
                )
                continue
            new_track = Track3D(
                detection,
                self.dt,
                self.max_age,
                self.min_hits,
                self.projection_matrix,
            )
            self.tracks.append(new_track)
            self.logger.debug(
                f"Created new Track ID {new_track.id} for Detection ID {detection.class_id}",
            )

        # Remove dead tracks
        before_removal = len(self.tracks)
        self.tracks = [track for track in self.tracks if track.time_since_update <= self.max_age]
        after_removal = len(self.tracks)
        removed_tracks = before_removal - after_removal
        if removed_tracks > 0:
            self.logger.info(f"Removed {removed_tracks} dead tracks.")

        # Prepare output tracked_objects
        tracked_objects = []
        for track in self.tracks:
            if not track.confirmed:
                continue
            position_3d = track.get_position()
            tracked_object = TrackedObject(
                track_id=track.id,
                bbox=[int(x) for x in track.bbox],
                confidence=track.confidence,
                class_id=track.class_id,
                label=track.label,
                dimensions=list(track.dimensions) if track.dimensions is not None else [],
                position_3d=position_3d.tolist(),
                mask=track.mask,
            )
            tracked_objects.append(tracked_object)
            self.logger.debug(
                f"Track {track.id}: {track.label} at {position_3d}. Updates: {track.time_since_update}",
            )
            self.logger.debug(
                f"Track ID: {track.id}, Label: {track.label}, Position: {position_3d}, Time since update: {track.time_since_update}",
            )
        self.logger.debug(f"Tracking {len(tracked_objects)} objects.")
        return tracked_objects

    def associate_detections_to_tracks(
        self,
        detections: list[DetectionResult],
        tracks: list[Track3D],
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        """
        Assign detections to tracks using a combined cost of position and appearance.

        Args:
            detections (list[DetectionResult]): list of current detections.
            tracks (list[Track3D]): list of existing tracks.

        Returns:
            tuple[list[tuple[int, int]], list[int], list[int]]: Matched pairs, unmatched tracks, unmatched detections.

        """
        if len(tracks) == 0:
            return [], [], list(range(len(detections)))
        if len(detections) == 0:
            return [], list(range(len(tracks))), []

        num_tracks = len(tracks)
        num_detections = len(detections)
        cost_matrix = np.zeros((num_tracks, num_detections), dtype=np.float32)

        for t_idx, track in enumerate(tracks):
            track_pos = track.get_position()
            track_feature = track.get_appearance_feature()
            for d_idx, detection in enumerate(detections):
                det_pos = np.array(detection.position_3d)
                position_distance = np.linalg.norm(track_pos - det_pos)
                # Appearance cost
                detection_feature = detection.appearance_feature
                if track_feature.size > 0 and detection_feature.size > 0:
                    # Normalize histograms
                    track_feature_norm = track_feature / (np.sum(track_feature) + 1e-6)
                    detection_feature_norm = detection_feature / (np.sum(detection_feature) + 1e-6)
                    # Compute Bhattacharyya distance
                    appearance_distance = cv2.compareHist(
                        track_feature_norm.astype("float32"),
                        detection_feature_norm.astype("float32"),
                        cv2.HISTCMP_BHATTACHARYYA,
                    )
                else:
                    appearance_distance = 1.0  # Maximum cost if features are missing
                # Combined cost
                cost = position_distance + 0.5 * appearance_distance
                cost_matrix[t_idx, d_idx] = cost

        self.logger.debug(f"Cost Matrix:\n{cost_matrix}")

        # Apply gating to limit associations
        max_cost = 200.0  # Maximum allowed cost for association
        cost_matrix[cost_matrix > max_cost] = max_cost + 1

        # Apply Hungarian algorithm
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        matches = []
        unmatched_tracks = list(range(num_tracks))
        unmatched_detections = list(range(num_detections))

        for r, c in zip(row_indices, col_indices, strict=False):
            if cost_matrix[r, c] > max_cost:
                continue
            matches.append((r, c))
            unmatched_tracks.remove(r)
            unmatched_detections.remove(c)

        self.logger.debug(f"Matched: {matches}")
        self.logger.debug(f"Unmatched Tracks: {unmatched_tracks}")
        self.logger.debug(f"Unmatched Detections: {unmatched_detections}")

        return matches, unmatched_tracks, unmatched_detections
