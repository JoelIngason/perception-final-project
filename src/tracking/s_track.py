import numpy as np
from ultralytics.trackers.basetrack import BaseTrack, TrackState

from .kalman import KalmanFilterXYZAH
from .utils.bbox_utils import xyzwh2tlzwh


class STrack(BaseTrack):
    """
    Single object tracking representation that uses Kalman filtering for state estimation.
    """

    shared_kalman = KalmanFilterXYZAH()

    _id = 0  # Class variable for unique track IDs

    def __init__(self, xyzwh, score, cls_label):
        super().__init__()
        # xyzwh+idx or xywhaz+idx
        assert len(xyzwh) in {5, 6}, f"expected 5 or 6 values but got {len(xyzwh)}"
        # Convert from [x, y, z, w, h, ...] or [x, y, z, w, h, a, ...]
        # Convert to [x1, y1, z, w, h] where (x1, y1) is top-left
        self._tlzwh = np.asarray(
            xyzwh2tlzwh(np.array(xyzwh[:5], dtype=np.float32)),
            dtype=np.float32,
        )  # Updated to include z
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.score = score
        self.tracklet_len = 0
        self.cls = cls_label
        # self.idx = xyzwh[-1]
        self.angle = xyzwh[5] if len(xyzwh) == 7 else None

    @classmethod
    def next_id(cls):
        cls._id += 1
        return cls._id

    def predict(self):
        """Predicts the next state (mean and covariance) of the object using the Kalman filter."""
        if self.mean is None or self.kalman_filter is None:
            return
        mean_state = self.mean.copy()
        # Zero the velocity of z and aspect ratio
        if self.state != TrackState.Tracked:
            mean_state[7:] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks):
        """Perform multi-object predictive tracking using Kalman filter for the provided list of STrack instances."""
        if len(stracks) <= 0:
            return
        multi_mean = np.asarray([st.mean.copy() for st in stracks])
        multi_covariance = np.asarray([st.covariance for st in stracks])
        for i, st in enumerate(stracks):
            if st.state != TrackState.Tracked:
                # Zero the z velocity and height and aspect ratio velocity
                multi_mean[i][7:] = 0

        multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(
            multi_mean,
            multi_covariance,
        )
        for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance, strict=False)):
            stracks[i].mean = mean
            stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id):
        """Activate a new tracklet using the provided Kalman filter and initialize its state and covariance."""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        # Include z in the conversion
        self.mean, self.covariance = self.kalman_filter.initiate(
            self.convert_coords(self._tlzwh),
        )

        assert self.mean.shape == (10,), f"Expected mean shape (10,), got {self.mean.shape}"
        assert self.covariance.shape == (
            10,
            10,
        ), f"Expected covariance shape (10,10), got {self.covariance.shape}"

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, measurement_mask=None, new_id=False):
        """Reactivates a previously lost track using new detection data and updates its state and attributes."""
        if self.kalman_filter is None:
            return
        # Include z in the conversion
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean,
            self.covariance,
            self.convert_coords(new_track.tlzwh),
            measurement_mask=measurement_mask,
        )
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score
        self.cls = new_track.cls
        self.angle = new_track.angle
        # self.idx = new_track.idx

    def update(self, new_track, frame_id, measurement_mask=None):
        """
        Update the state of a matched track.
        """
        if self.kalman_filter is None:
            return
        self.frame_id = frame_id
        self.tracklet_len += 1

        new_tlzwh = new_track.tlzwh
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean,
            self.covariance,
            self.convert_coords(new_tlzwh),
            measurement_mask=measurement_mask,
        )
        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score
        self.cls = new_track.cls
        self.angle = new_track.angle
        # self.idx = new_track.idx

    def convert_coords(self, tlzwh):
        """
        Convert a bounding box's top-left-width-height format to its x-y-z-aspect-height or x-y-z-width-height equivalent.
        """
        return self.tlzwh_to_xyzah(tlzwh)

    @property
    def tlzwh(self):
        """Returns the bounding box in top-left-width-height format from the current state estimate."""
        if self.mean is None:
            return self._tlzwh.copy()
        x, y, z, a, h = self.mean[:5].copy()  # Correctly unpack [x, y, z, a, h]
        w = a * h  # Compute width from aspect ratio and height
        ret = np.array([x - w / 2, y - h / 2, z, w, h], dtype=np.float32)  # [x1, y1, z, w, h]
        assert len(ret) == 5, f"Expected 5 values, got {len(ret)}"
        return ret

    @property
    def xyzxy(self):
        """Converts bounding box to [x1, y1, z, x2, y2] format."""
        ret = self.tlzwh.copy()  # [x1, y1, z, w, h]
        x1, y1, z, w, h = ret
        x2 = x1 + w
        y2 = y1 + h
        ret_xyzxy = np.array([x1, y1, z, x2, y2], dtype=np.float32)  # [x1, y1, z, x2, y2]
        return ret_xyzxy

    @property
    def xyxy(self):
        """Converts bounding box from (top left x, top left y, z, width, height) to (min x, min y, max x, max y) format."""
        ret = self.tlzwh.copy()  # [x1, y1, z, w, h]
        x1, y1, z, w, h = ret
        x2 = x1 + w
        y2 = y1 + h
        ret_xyxy = np.array([x1, y1, x2, y2], dtype=np.float32)  # [x1, y1, x2, y2]
        return ret_xyxy

    @staticmethod
    def tlzwh_to_xyzah(tlzwh):
        """
        Convert bounding box from [x1, y1, z, w, h] format to [x, y, z, a, h] format.
        """
        ret = np.asarray(tlzwh).copy()
        ret[:2] += ret[3:] / 2  # Convert to center x, y
        ret[3] /= ret[4]  # Aspect ratio a = w / h
        return ret

    @property
    def xywh(self):
        """Returns the current position of the bounding box in (center x, center y, width, height) format."""
        ret = np.asarray(self.tlzwh).copy()
        ret[:2] += ret[3:] / 2
        # Remove z
        ret = np.delete(ret, 2)
        return ret

    @property
    def xyzwh(self):
        """Returns the current position of the bounding box in (center x, center y, center z, width, height) format."""
        ret = np.asarray(self.tlzwh).copy()
        ret[:2] += ret[3:] / 2
        return ret

    @property
    def result(self):
        """Returns the current tracking results in the appropriate bounding box format."""
        coords = self.xyzxy
        return coords.tolist() + [self.track_id, self.score, self.cls]

    def __repr__(self):
        """Return a string representation of the STrack object including start frame, end frame, and track ID."""
        return f"OT_{self.track_id}_({self.start_frame}-{self.frame_id})"

    def mark_lost(self):
        """Mark the track as lost."""
        self.state = TrackState.Lost

    def mark_removed(self):
        """Mark the track as removed."""
        self.state = TrackState.Removed

    @staticmethod
    def reset_id():
        """Reset the class variable _id."""
        STrack._id = 0


class STrackFeature(STrack):
    """
    Extended STrack with feature embedding for appearance-based tracking.
    """

    def __init__(self, xyzwh, score, cls_label, feature=None):
        super().__init__(xyzwh, score, cls_label)
        self.feature = feature  # Deep feature embedding
        self.history = {}  # To store history for debugging
        self.updated_in_frame = False  # Flag to prevent multiple updates per frame

    def update(self, new_track, frame_id, measurement_mask=None):
        if self.updated_in_frame:
            print(f"Warning: Track {self.track_id} updated multiple times in frame {frame_id}")
        super().update(new_track, frame_id, measurement_mask)
        self.feature = new_track.feature  # Update feature
        self.history[frame_id] = self.tlzwh  # Store history
        self.updated_in_frame = True  # Set the flag

    def re_activate(self, new_track, frame_id, measurement_mask=None, new_id=False):
        if self.updated_in_frame:
            print(f"Warning: Track {self.track_id} updated multiple times in frame {frame_id}")
        super().re_activate(new_track, frame_id, measurement_mask, new_id)
        self.feature = new_track.feature  # Update feature
        self.history[frame_id] = self.tlzwh  # Store history

    def get_cached_feature(self):
        """
        Returns the cached feature for the track.
        """
        return self.feature

    def compute_average_velocity(self):
        """
        Compute the average velocity of the track based on its history.

        Returns:
            np.ndarray: Average velocity vector [vx, vy, vz, va, vh].

        """
        if len(self.history) < 2:
            return np.zeros(5, dtype=np.float32)

        sorted_frames = sorted(self.history.keys())
        deltas = []
        for i in range(1, len(sorted_frames)):
            frame_diff = sorted_frames[i] - sorted_frames[i - 1]
            if frame_diff == 0:
                continue
            prev_bbox = self.history[sorted_frames[i - 1]]
            curr_bbox = self.history[sorted_frames[i]]
            delta = (curr_bbox[:2] - prev_bbox[:2]) / frame_diff  # [dx, dy]
            delta_z = (curr_bbox[2] - prev_bbox[2]) / frame_diff  # dz
            delta_a = (self.mean[3] - (prev_bbox[3] / prev_bbox[4])) / frame_diff  # da
            delta_h = (self.mean[4] - prev_bbox[4]) / frame_diff  # dh
            deltas.append([delta[0], delta[1], delta_z, delta_a, delta_h])

        if not deltas:
            return np.zeros(5, dtype=np.float32)

        deltas = np.array(deltas)
        average_velocity = np.mean(deltas, axis=0)
        return average_velocity
