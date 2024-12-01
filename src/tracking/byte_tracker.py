import argparse
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial.distance import cdist
from ultralytics.trackers.basetrack import TrackState

from src.models.feature_extractor.feature_extractor import FeatureExtractor

from .kalman import KalmanFilterXYZAH
from .s_track import STrackFeature
from .utils.assignment import linear_assignment
from .utils.bbox_utils import crop_bbox
from .utils.distance import iou_distance


class BYTETracker:
    """
    BYTETracker: A tracking algorithm built on top of YOLOv8 for object detection and tracking.
    """

    def __init__(
        self,
        config_path: str | argparse.Namespace = "config/byte_tracker.yaml",
        frame_rate: int = 30,
        feature_extractor: FeatureExtractor | None = None,
    ):
        """
        Initialize the BYTETracker.

        Args:
            config_path (str): Path to the BYTETracker configuration YAML file.
            frame_rate (int): Frame rate of the video/input.
            feature_extractor (Optional): Instance of a feature extractor.

        """
        if isinstance(config_path, str):
            self.args = self.load_config(config_path)
        else:
            self.args = config_path
        self.tracked_stracks: list[STrackFeature] = []
        self.lost_stracks: list[STrackFeature] = []
        self.removed_stracks: list[STrackFeature] = []

        self.frame_id: int = 0
        self.max_time_lost: int = int(frame_rate / 10.0 * self.args.track_buffer)
        self.kalman_filter = self.get_kalmanfilter()
        self.reset_id()
        self.feature_extractor = feature_extractor  # Instance of FeatureExtractor

        # Define maximum average velocity thresholds (pixels per frame)
        self.max_avg_velocity = {
            "x": 50,
            "y": 50,
        }

    @staticmethod
    def load_config(config_path: str) -> argparse.Namespace:
        """
        Load tracker configuration from a YAML file.

        Args:
            config_path (str): Path to the configuration YAML file.

        Returns:
            argparse.Namespace: Configuration parameters.

        """
        config_file = Path(config_path)
        if not config_file.is_file():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with config_file.open("r") as f:
            args_byte_tracker = yaml.safe_load(f)

        return argparse.Namespace(**args_byte_tracker)

    def update(
        self,
        detections: np.ndarray,
        measurement_masks: np.ndarray,
        img: np.ndarray,
    ) -> np.ndarray:
        """
        Update tracker with new detections and corresponding measurement masks.

        Args:
            detections (np.ndarray): Array of detections with shape (N, 7).
            measurement_masks (np.ndarray): Array of measurement masks.
            img (np.ndarray): Current frame image.

        Returns:
            np.ndarray: Array of activated track results.

        """
        self.frame_id += 1
        activated_stracks: list[STrackFeature] = []
        refind_stracks: list[STrackFeature] = []
        lost_stracks: list[STrackFeature] = []
        removed_stracks: list[STrackFeature] = []

        if detections.size == 0:
            self.handle_no_detections()
            return self.get_active_tracks()

        # Process detections
        bboxes, scores, cls = self.extract_detection_components(detections)
        detections, scores_keep, cls_keep, dets_second, scores_second, cls_second = (
            self.split_detections(bboxes, scores, cls)
        )

        # Extract features for high-score detections
        features_keep = self.extract_features(dets_keep=detections, img=img)

        detections_high = self.init_track(
            dets=detections,
            scores=scores_keep,
            cls_labels=cls_keep,
            features=features_keep,
        )

        # Separate confirmed and unconfirmed tracks
        unconfirmed, tracked_stracks = self.separate_stracks()

        # Step 1: First association with high score detections
        strack_pool = self.joint_stracks(tracked_stracks, self.lost_stracks)
        self.multi_predict(strack_pool)
        dists = self.get_dists(strack_pool, detections_high)
        matches, u_track, u_detection = linear_assignment(
            dists,
            thresh=self.args.match_thresh,
        )

        # Process matches
        u_track, u_detection, activated_stracks, refind_stracks = self.process_matches(
            matches,
            strack_pool,
            detections_high,
            measurement_masks,
            u_track,
            u_detection,
            activated_stracks,
            refind_stracks,
        )

        # Step 2: Second association with low score detections
        if self.args.fuse_score and self.feature_extractor:
            (
                activated_stracks,
                refind_stracks,
                lost_stracks,
            ) = self.second_association(
                tracked_stracks,
                dets_second,
                scores_second,
                cls_second,
                img,
                measurement_masks,
                activated_stracks,
                refind_stracks,
            )

        # Step 3: Associate unconfirmed tracks with remaining detections
        (
            u_unconfirmed,
            u_detection_final,
            activated_stracks,
            removed_stracks,
        ) = self.associate_unconfirmed_tracks(
            detections_high,
            unconfirmed,
            u_detection,
            measurement_masks,
            activated_stracks,
            removed_stracks,
        )

        # Step 4: Activate new tracks for unmatched detections
        activated_stracks = self.activate_new_tracks(
            detections_high,
            u_detection,
            activated_stracks,
        )

        # Step 5: Mark tracks as removed if they have been lost for too long
        removed_stracks = self.mark_lost_tracks(removed_stracks)

        # Update track lists
        self.update_track_lists(activated_stracks, refind_stracks, lost_stracks, removed_stracks)

        return self.get_active_tracks()

    def handle_no_detections(self):
        """Handle the scenario where there are no detections in the current frame."""
        dists = np.zeros((len(self.tracked_stracks), 0), dtype=np.float32)
        matches, u_track, u_detection = linear_assignment(dists, thresh=self.args.match_thresh)
        # No detections to process
        # Further handling can be implemented as needed

    def get_active_tracks(self) -> np.ndarray:
        """
        Retrieve all active and activated tracks.

        Returns:
            np.ndarray: Array of active track results.

        """
        return np.asarray(
            [track.result for track in self.tracked_stracks if track.is_activated],
            dtype=np.float32,
        )

    def extract_detection_components(
        self,
        detections: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract bounding boxes, scores, and class labels from detections.

        Args:
            detections (np.ndarray): Array of detections.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Bounding boxes, scores, and class labels.

        """
        bboxes = detections[:, :5]  # [x, y, z, a, h]
        # Add index column to bboxes
        bboxes = np.concatenate([bboxes, np.arange(len(bboxes)).reshape(-1, 1)], axis=-1)
        scores = detections[:, 5]
        cls = detections[:, 6] if detections.shape[1] > 6 else np.zeros(len(bboxes))
        return bboxes, scores, cls

    def split_detections(
        self,
        bboxes: np.ndarray,
        scores: np.ndarray,
        cls: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split detections into high and low score groups.

        Args:
            bboxes (np.ndarray): Bounding boxes.
            scores (np.ndarray): Detection scores.
            cls (np.ndarray): Class labels.

        Returns:
            Tuple containing:
                - High score detections.
                - High score scores.
                - High score class labels.
                - Second group detections.
                - Second group scores.
                - Second group class labels.

        """
        remain_inds = scores >= self.args.track_high_thresh
        inds_low = scores > self.args.track_low_thresh
        inds_high = scores < self.args.track_high_thresh

        inds_second = inds_low & inds_high
        dets_second = bboxes[inds_second]
        detections = bboxes[remain_inds]
        scores_keep = scores[remain_inds]
        scores_second = scores[inds_second]
        cls_keep = cls[remain_inds]
        cls_second = cls[inds_second]

        return detections, scores_keep, cls_keep, dets_second, scores_second, cls_second

    def extract_features(self, dets_keep: np.ndarray, img: np.ndarray) -> np.ndarray | None:
        """
        Extract features for high-score detections.

        Args:
            dets_keep (np.ndarray): High-score detections.
            img (np.ndarray): Current frame image.

        Returns:
            Optional[np.ndarray]: Extracted features or None.

        """
        cropped_imgs = [crop_bbox(img, det) for det in dets_keep]
        if self.feature_extractor:
            return self.feature_extractor(cropped_imgs)
        return None

    def separate_stracks(
        self,
    ) -> tuple[list[STrackFeature], list[STrackFeature]]:
        """
        Separate tracks into unconfirmed and confirmed (tracked) stracks.

        Returns:
            Tuple[List[STrackFeature], List[STrackFeature]]: Unconfirmed and confirmed stracks.

        """
        unconfirmed = [track for track in self.tracked_stracks if not track.is_activated]
        tracked_stracks = [track for track in self.tracked_stracks if track.is_activated]
        return unconfirmed, tracked_stracks

    def process_matches(
        self,
        matches: list[tuple[int, int]],
        strack_pool: list[STrackFeature],
        detections_high: list[STrackFeature],
        measurement_masks: np.ndarray,
        u_track: np.ndarray,
        u_detection: np.ndarray,
        activated_stracks: list[STrackFeature],
        refind_stracks: list[STrackFeature],
    ) -> tuple[np.ndarray, np.ndarray, list[STrackFeature], list[STrackFeature]]:
        """
        Process matched tracks and detections.

        Args:
            matches (List[Tuple[int, int]]): List of matched track and detection indices.
            strack_pool (List[STrackFeature]): Pool of tracks available for matching.
            detections_high (List[STrackFeature]): High-score detections.
            measurement_masks (np.ndarray): Measurement masks for detections.
            u_track (np.ndarray): Unmatched track indices.
            u_detection (np.ndarray): Unmatched detection indices.
            activated_stracks (List[STrackFeature]): List to store activated stracks.
            refind_stracks (List[STrackFeature]): List to store refound stracks.

        Returns:
            Updated u_track, u_detection, activated_stracks, refind_stracks.

        """
        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections_high[idet]
            mask = measurement_masks[idet]

            # Enforce velocity constraints before updating
            avg_velocity = track.compute_average_velocity()
            if (
                abs(avg_velocity[0]) > self.max_avg_velocity["x"]
                or abs(avg_velocity[1]) > self.max_avg_velocity["y"]
            ):
                # Velocity too high; do not match
                u_track = np.append(u_track, itracked)
                u_detection = np.append(u_detection, idet)
                continue

            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id, measurement_mask=mask)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id, measurement_mask=mask, new_id=False)
                refind_stracks.append(track)

        return u_track, u_detection, activated_stracks, refind_stracks

    def second_association(
        self,
        tracked_stracks: list[STrackFeature],
        dets_second: np.ndarray,
        scores_second: np.ndarray,
        cls_second: np.ndarray,
        img: np.ndarray,
        measurement_masks: np.ndarray,
        activated_stracks: list[STrackFeature],
        refind_stracks: list[STrackFeature],
    ) -> tuple[
        list[STrackFeature],
        list[STrackFeature],
        list[STrackFeature],
    ]:
        """
        Perform second association with low score detections.

        Args:
            tracked_stracks (list[STrackFeature]): Currently tracked stracks.
            dets_second (np.ndarray): Low score detections.
            scores_second (np.ndarray): Low score detection scores.
            cls_second (np.ndarray): Low score detection class labels.
            img (np.ndarray): Current frame image.
            measurement_masks (np.ndarray): Measurement masks for detections.
            activated_stracks (list[STrackFeature]): List to store activated stracks.
            refind_stracks (list[STrackFeature]): List to store refound stracks.

        Returns:
            activated_stracks, refind_stracks, lost_stracks.

        """
        # Extract features for second group
        features_second = self.extract_features(dets_keep=dets_second, img=img)

        detections_second = self.init_track(
            dets=dets_second,
            scores=scores_second,
            cls_labels=cls_second,
            features=features_second,
        )

        dists_second = iou_distance(tracked_stracks, detections_second)

        if (
            self.args.fuse_score
            and self.feature_extractor
            and (
                len(tracked_stracks) > 0
                and len(detections_second) > 0
                and all(track.feature is not None for track in tracked_stracks)
                and all(det.feature is not None for det in detections_second)
            )
        ):
            appearance_dists = cdist(
                np.array([track.feature for track in tracked_stracks]),
                np.array([det.feature for det in detections_second]),
                metric="cosine",
            )
            alpha = 0.3  # Weight for IoU
            beta = 0.7  # Weight for appearance
            dists_second = alpha * dists_second + beta * appearance_dists

        # Ensure dists_second is not empty before assignment
        if dists_second.size > 0:
            matches_second, u_track_second, u_detection_second = linear_assignment(
                dists_second,
                thresh=0.5,
            )

            for itracked, idet in matches_second:
                track = tracked_stracks[itracked]
                det = detections_second[idet]
                mask = measurement_masks[idet]

                # Enforce velocity constraints before updating
                avg_velocity = track.compute_average_velocity()
                if (
                    abs(avg_velocity[0]) > self.max_avg_velocity["x"]
                    or abs(avg_velocity[1]) > self.max_avg_velocity["y"]
                ):
                    # Velocity too high; do not match
                    u_track_second = np.append(u_track_second, itracked)
                    u_detection_second = np.append(u_detection_second, idet)
                    continue

                if track.state == TrackState.Tracked:
                    track.update(det, self.frame_id, measurement_mask=mask)
                    activated_stracks.append(track)
                else:
                    track.re_activate(det, self.frame_id, measurement_mask=mask, new_id=False)
                    refind_stracks.append(track)

            # Handle unmatched tracks after second association
            lost_stracks = []
            for it in u_track_second:
                track = tracked_stracks[it]
                if track.state != TrackState.Lost:
                    track.mark_lost()
                    lost_stracks.append(track)

            return (
                activated_stracks,
                refind_stracks,
                lost_stracks,
            )

        return activated_stracks, refind_stracks, []

    def associate_unconfirmed_tracks(
        self,
        detections_high: list[STrackFeature],
        unconfirmed: list[STrackFeature],
        u_detection: np.ndarray,
        measurement_masks: np.ndarray,
        activated_stracks: list[STrackFeature],
        removed_stracks: list[STrackFeature],
    ) -> tuple[np.ndarray, np.ndarray, list[STrackFeature], list[STrackFeature]]:
        """
        Associate unconfirmed tracks with remaining detections.

        Args:
            detections_high (List[STrackFeature]): High-score detections.
            unconfirmed (List[STrackFeature]): List of unconfirmed tracks.
            u_detection (np.ndarray): Unmatched detection indices.
            measurement_masks (np.ndarray): Measurement masks for detections.
            activated_stracks (List[STrackFeature]): List to store activated stracks.
            removed_stracks (List[STrackFeature]): List to store removed stracks.

        Returns:
            Tuple of updated unconfirmed tracks, unmatched detections, activated stracks, and removed stracks.

        """
        # Assuming detections_high is accessible; if not, pass it as a parameter
        detections_unconfirmed = [detections_high[i] for i in u_detection]
        masks_unconfirmed = [measurement_masks[i] for i in u_detection]
        dists_unconfirmed = self.get_dists(unconfirmed, detections_unconfirmed)
        u_unconfirmed = np.array([], dtype=int)
        u_detection_final = np.array([], dtype=int)
        if dists_unconfirmed.size > 0:
            matches_unconfirmed, u_unconfirmed, u_detection_final = linear_assignment(
                dists_unconfirmed,
                thresh=0.7,
            )

            for itracked, idet in matches_unconfirmed:
                track = unconfirmed[itracked]
                det = detections_unconfirmed[idet]
                mask = masks_unconfirmed[idet]

                # Enforce velocity constraints before updating
                avg_velocity = track.compute_average_velocity()
                if (
                    abs(avg_velocity[0]) > self.max_avg_velocity["x"]
                    or abs(avg_velocity[1]) > self.max_avg_velocity["y"]
                ):
                    # Velocity too high; do not match
                    u_unconfirmed = np.append(u_unconfirmed, itracked)
                    u_detection_final = np.append(u_detection_final, idet)
                    continue

                track.update(
                    det,
                    self.frame_id,
                    measurement_mask=mask,
                )
                activated_stracks.append(track)

            for it in u_unconfirmed:
                track = unconfirmed[it]
                track.mark_removed()
                removed_stracks.append(track)

        return u_unconfirmed, u_detection_final, activated_stracks, removed_stracks

    def activate_new_tracks(
        self,
        detections_high: list[STrackFeature],
        u_detection: np.ndarray,
        activated_stracks: list[STrackFeature],
    ) -> list[STrackFeature]:
        """
        Activate new tracks for unmatched detections_high.

        Args:
            detections_high (List[STrackFeature]): High-score detections.
            u_detection (np.ndarray): Unmatched detection indices.
            activated_stracks (List[STrackFeature]): List to store activated stracks.

        Returns:
            List[STrackFeature]: Activated stracks.

        """
        for idet in u_detection:
            track = detections_high[idet]
            if track.score < self.args.new_track_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_stracks.append(track)

        return activated_stracks

    def mark_lost_tracks(self, removed_stracks: list[STrackFeature]) -> list[STrackFeature]:
        """
        Mark tracks as removed if they have been lost for too long.

        Args:
            removed_stracks (list[STrackFeature]): List to store removed stracks.

        Returns:
            List[STrackFeature]: Removed stracks.

        """
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        return removed_stracks

    def update_track_lists(
        self,
        activated_stracks: list[STrackFeature],
        refind_stracks: list[STrackFeature],
        lost_stracks: list[STrackFeature],
        removed_stracks: list[STrackFeature],
    ):
        """
        Update the track lists with activated, refound, lost, and removed stracks.

        Args:
            activated_stracks (List[STrackFeature]): Activated stracks.
            refind_stracks (List[STrackFeature]): Refound stracks.
            lost_stracks (List[STrackFeature]): Lost stracks.
            removed_stracks (List[STrackFeature]): Removed stracks.

        """
        # Update track lists
        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = self.joint_stracks(self.tracked_stracks, activated_stracks)
        self.tracked_stracks = self.joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = self.sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = self.sub_stracks(self.lost_stracks, self.removed_stracks)
        self.tracked_stracks, self.lost_stracks = self.remove_duplicate_stracks(
            self.tracked_stracks,
            self.lost_stracks,
        )
        self.removed_stracks.extend(removed_stracks)
        if len(self.removed_stracks) > 1000:
            self.removed_stracks = self.removed_stracks[-999:]

    def init_track(
        self,
        dets: np.ndarray,
        scores: np.ndarray,
        cls_labels: np.ndarray,
        features: np.ndarray | None,
    ) -> list[STrackFeature]:
        """
        Initialize object tracking with given detections, scores, class labels, and features.

        Args:
            dets (np.ndarray): Detections.
            scores (np.ndarray): Detection scores.
            cls_labels (np.ndarray): Class labels.
            features (Optional[np.ndarray]): Extracted features.

        Returns:
            List[STrackFeature]: Initialized track features.

        """
        if len(dets) == 0:
            return []

        if features is not None:
            # Ensure features length matches detections
            if len(features) != len(dets):
                raise ValueError("Number of features does not match number of detections.")
            feature_iter = iter(features)
        else:
            # If no features, use None for all
            feature_iter = iter([None] * len(dets))

        return [
            STrackFeature(xyzwh, s, c, feature=f)
            for xyzwh, s, c, f in zip(dets, scores, cls_labels, feature_iter, strict=False)
        ]

    def get_dists(self, tracks: list[STrackFeature], detections: list[STrackFeature]) -> np.ndarray:
        """
        Calculate the distance between tracks and detections using both IoU and appearance features.

        Args:
            tracks (List[STrackFeature]): List of tracks.
            detections (List[STrackFeature]): List of detections.

        Returns:
            np.ndarray: Combined distance matrix.

        """
        iou_dists = iou_distance(tracks, detections)

        # Compute appearance-based distances (Cosine distance)
        track_features = np.array([track.feature for track in tracks if track.feature is not None])
        detection_features = np.array(
            [det.feature for det in detections if det.feature is not None],
        )

        if track_features.size > 0 and detection_features.size > 0:
            appearance_dists = cdist(track_features, detection_features, metric="cosine")
        else:
            appearance_dists = np.ones((len(tracks), len(detections)), dtype=np.float32)

        # Combine distances
        alpha = 0.5  # Weight for IoU
        beta = 0.5  # Weight for appearance
        combined_dists = alpha * iou_dists + beta * appearance_dists

        return combined_dists

    def multi_predict(self, tracks: list[STrackFeature]):
        """
        Predict the next states for multiple tracks using Kalman filter.

        Args:
            tracks (List[STrackFeature]): List of tracks to predict.

        """
        STrackFeature.multi_predict(tracks)

    def get_kalmanfilter(self) -> KalmanFilterXYZAH:
        """
        Return a Kalman filter object for tracking bounding boxes using KalmanFilterXYZAH.

        Returns:
            KalmanFilterXYZAH: Instance of KalmanFilterXYZAH.

        """
        return KalmanFilterXYZAH()

    def reset_id(self):
        """
        Reset the ID counter for STrack instances to ensure unique track IDs across tracking sessions.
        """
        STrackFeature.reset_id()

    def reset(self):
        """
        Reset the tracker by clearing all tracked, lost, and removed tracks and reinitializing the Kalman filter.
        """
        self.tracked_stracks = []
        self.lost_stracks = []
        self.removed_stracks = []
        self.frame_id = 0
        self.kalman_filter = self.get_kalmanfilter()
        self.reset_id()

    @staticmethod
    def joint_stracks(
        tlista: list[STrackFeature],
        tlistb: list[STrackFeature],
    ) -> list[STrackFeature]:
        """
        Combine two lists of STrackFeature objects into a single list, ensuring no duplicates based on track IDs.

        Args:
            tlista (List[STrackFeature]): First list of tracks.
            tlistb (List[STrackFeature]): Second list of tracks.

        Returns:
            List[STrackFeature]: Combined list of tracks.

        """
        exists = {t.track_id: True for t in tlista}
        for t in tlistb:
            if t.track_id not in exists:
                tlista.append(t)
                exists[t.track_id] = True
        return tlista

    @staticmethod
    def sub_stracks(
        tlista: list[STrackFeature],
        tlistb: list[STrackFeature],
    ) -> list[STrackFeature]:
        """
        Filter out the stracks present in the second list from the first list.

        Args:
            tlista (List[STrackFeature]): First list of tracks.
            tlistb (List[STrackFeature]): Second list of tracks to remove.

        Returns:
            List[STrackFeature]: Filtered list of tracks.

        """
        track_ids_b = {t.track_id for t in tlistb}
        return [t for t in tlista if t.track_id not in track_ids_b]

    @staticmethod
    def remove_duplicate_stracks(
        stracksa: list[STrackFeature],
        stracksb: list[STrackFeature],
    ) -> tuple[list[STrackFeature], list[STrackFeature]]:
        """
        Remove duplicate stracks from two lists based on Intersection over Union (IoU) distance.

        Args:
            stracksa (list[STrackFeature]): First list of tracks.
            stracksb (list[STrackFeature]): Second list of tracks.

        Returns:
            Tuple[list[STrackFeature], list[STrackFeature]]: Lists after removing duplicates.

        """
        pdist = iou_distance(stracksa, stracksb)
        pairs = np.where(pdist < 0.15)
        dupa, dupb = set(), set()
        for p, q in zip(*pairs, strict=False):
            if p in dupa or q in dupb:
                continue
            timep = stracksa[p].frame_id - stracksa[p].start_frame
            timeq = stracksb[q].frame_id - stracksb[q].start_frame
            if timep > timeq:
                dupb.add(q)
            else:
                dupa.add(p)
        resa = [t for i, t in enumerate(stracksa) if i not in dupa]
        resb = [t for i, t in enumerate(stracksb) if i not in dupb]
        return resa, resb
