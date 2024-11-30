import numpy as np
from scipy.spatial.distance import cdist
from ultralytics.trackers.basetrack import TrackState

from .kalman import KalmanFilterXYZAH
from .s_track import STrack, STrackFeature
from .utils.assignment import linear_assignment
from .utils.bbox_utils import crop_bbox
from .utils.distance import iou_distance


class BYTETracker:
    """
    BYTETracker: A tracking algorithm built on top of YOLOv8 for object detection and tracking.
    """

    def __init__(self, args, frame_rate=30, feature_extractor=None):
        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]

        self.frame_id = 0
        self.args = args
        self.max_time_lost = int(frame_rate / 10.0 * args.track_buffer)
        self.kalman_filter = self.get_kalmanfilter()
        self.reset_id()
        self.feature_extractor = feature_extractor  # Instance of FeatureExtractor

        # Define maximum average velocity thresholds (pixels per frame)
        self.max_avg_velocity = {
            "x": 50,
            "y": 50,
        }

    def update(self, detections, measurement_masks, img):
        """
        Update tracker with new detections and corresponding measurement masks.
        """
        self.frame_id += 1
        activated_stracks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        if len(detections) == 0:
            dists = np.zeros((len(self.tracked_stracks), 0), dtype=np.float32)
            matches, u_track, u_detection = linear_assignment(dists, thresh=self.args.match_thresh)
            # No detections to process
            # Further handling can be implemented as needed
            return np.asarray(
                [x.result for x in self.tracked_stracks if x.is_activated],
                dtype=np.float32,
            )

        # Process detections
        bboxes = detections[:, :5]  # [x, y, z, a, h]
        scores = detections[:, 5]
        cls = detections[:, 6] if detections.shape[1] > 6 else np.zeros(len(bboxes))

        # Add index
        bboxes = np.concatenate([bboxes, np.arange(len(bboxes)).reshape(-1, 1)], axis=-1)

        # Split detections based on score thresholds
        remain_inds = scores >= self.args.track_high_thresh
        inds_low = scores > self.args.track_low_thresh
        inds_high = scores < self.args.track_high_thresh

        inds_second = inds_low & inds_high
        dets_second = bboxes[inds_second]
        dets = bboxes[remain_inds]
        scores_keep = scores[remain_inds]
        scores_second = scores[inds_second]
        cls_keep = cls[remain_inds]
        cls_second = cls[inds_second]

        # Extract features for high-score detections
        cropped_imgs = [crop_bbox(img, det) for det in dets]
        features_keep = self.feature_extractor(cropped_imgs) if self.feature_extractor else None

        detections_high = self.init_track(dets, scores_keep, cls_keep, features_keep, img)

        # Add newly detected tracklets to tracked_stracks
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        # Step 2: First association with high score detections
        strack_pool = self.joint_stracks(tracked_stracks, self.lost_stracks)
        # Predict the current location with KF
        self.multi_predict(strack_pool)

        # Calculate distances for association
        dists = self.get_dists(strack_pool, detections_high)
        matches, u_track, u_detection = linear_assignment(
            dists,
            thresh=self.args.match_thresh,
        )

        # Process matches
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

        # Step 3: Second association with low score detections
        if self.args.fuse_score and self.feature_extractor:
            # Only perform second association if feature extractor is available
            cropped_imgs_second = [crop_bbox(img, det) for det in dets_second]
            features_second = (
                self.feature_extractor(cropped_imgs_second) if self.feature_extractor else None
            )
            detections_second = self.init_track(
                dets_second,
                scores_second,
                cls_second,
                features_second,
                img,
            )

            dists_second = iou_distance(tracked_stracks, detections_second)
            if self.args.fuse_score and self.feature_extractor:
                if (
                    len(tracked_stracks) > 0
                    and len(detections_second) > 0
                    and all(track.feature is not None for track in tracked_stracks)
                    and all(det.feature is not None for det in detections_second)
                ):
                    appearance_dists = cdist(
                        np.array([track.feature for track in tracked_stracks]),
                        np.array([det.feature for det in detections_second]),
                        metric="cosine",
                    )
                    alpha = 0.5  # Weight for IoU
                    beta = 0.5  # Weight for appearance
                    dists_second = alpha * dists_second + beta * appearance_dists
                else:
                    # If features are missing, skip combining distances
                    pass

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
                for it in u_track_second:
                    track = tracked_stracks[it]
                    if track.state != TrackState.Lost:
                        track.mark_lost()
                        lost_stracks.append(track)

        # Step 4: Associate unconfirmed tracks with remaining detections
        detections_unconfirmed = [detections_high[i] for i in u_detection]
        masks_unconfirmed = [measurement_masks[i] for i in u_detection]
        dists_unconfirmed = self.get_dists(unconfirmed, detections_unconfirmed)
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

        # Step 5: Activate new tracks for unmatched detections
        for inew in u_detection_final:
            track = detections_unconfirmed[inew]
            if track.score < self.args.new_track_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_stracks.append(track)

        # Step 6: Mark tracks as removed if they have been lost for too long
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

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

        return np.asarray(
            [x.result for x in self.tracked_stracks if x.is_activated],
            dtype=np.float32,
        )

    def get_kalmanfilter(self):
        """Return a Kalman filter object for tracking bounding boxes using KalmanFilterXYZAH."""
        return KalmanFilterXYZAH()

    def init_track(self, dets, scores, cls_labels, features, img=None):
        """
        Initialize object tracking with given detections, scores, class labels, and features.
        """
        return (
            [
                STrackFeature(xyzwh, s, c, feature=f)
                for (xyzwh, s, c, f) in zip(dets, scores, cls_labels, features, strict=False)
            ]
            if len(dets)
            else []
        )

    def get_dists(self, tracks, detections):
        """
        Calculate the distance between tracks and detections using both IoU and appearance features.
        """
        # Compute IoU-based distances
        iou_dists = iou_distance(tracks, detections)

        # Compute appearance-based distances (Cosine distance)
        track_features = np.array([track.feature for track in tracks if track.feature is not None])
        detection_features = np.array(
            [det.feature for det in detections if det.feature is not None],
        )

        if len(track_features) > 0 and len(detection_features) > 0:
            appearance_dists = cdist(track_features, detection_features, metric="cosine")
        else:
            appearance_dists = np.ones((len(tracks), len(detections)))

        # Combine distances
        alpha = 0.5  # Weight for IoU
        beta = 0.5  # Weight for appearance
        combined_dists = alpha * iou_dists + beta * appearance_dists

        return combined_dists

    def multi_predict(self, tracks):
        """Predict the next states for multiple tracks using Kalman filter."""
        STrack.multi_predict(tracks)

    def reset_id(self):
        """Reset the ID counter for STrack instances to ensure unique track IDs across tracking sessions."""
        STrack.reset_id()

    def reset(self):
        """Reset the tracker by clearing all tracked, lost, and removed tracks and reinitializing the Kalman filter."""
        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]
        self.frame_id = 0
        self.kalman_filter = self.get_kalmanfilter()
        self.reset_id()

    @staticmethod
    def joint_stracks(tlista, tlistb):
        """
        Combine two lists of STrack objects into a single list, ensuring no duplicates based on track IDs.
        """
        exists = {}
        res = []
        for t in tlista:
            exists[t.track_id] = 1
            res.append(t)
        for t in tlistb:
            tid = t.track_id
            if not exists.get(tid, 0):
                exists[tid] = 1
                res.append(t)
        return res

    @staticmethod
    def sub_stracks(tlista, tlistb):
        """
        Filter out the stracks present in the second list from the first list.
        """
        track_ids_b = {t.track_id for t in tlistb}
        return [t for t in tlista if t.track_id not in track_ids_b]

    @staticmethod
    def remove_duplicate_stracks(stracksa, stracksb):
        """
        Remove duplicate stracks from two lists based on Intersection over Union (IoU) distance.
        """
        pdist = iou_distance(stracksa, stracksb)
        pairs = np.where(pdist < 0.15)
        dupa, dupb = [], []
        for p, q in zip(*pairs, strict=False):
            timep = stracksa[p].frame_id - stracksa[p].start_frame
            timeq = stracksb[q].frame_id - stracksb[q].start_frame
            if timep > timeq:
                dupb.append(q)
            else:
                dupa.append(p)
        resa = [t for i, t in enumerate(stracksa) if i not in dupa]
        resb = [t for i, t in enumerate(stracksb) if i not in dupb]
        return resa, resb
