import logging
from collections import defaultdict

import numpy as np
from scipy.optimize import linear_sum_assignment

from src.data_loader.dataset import TrackedObject
from src.utils.label_parser import GroundTruthObject


class Evaluator:
    """Evaluate the performance of the object tracking system."""

    def __init__(self, distance_threshold: float = 200.0):
        """Initialize the Evaluator for computing and reporting metrics."""
        self.logger = logging.getLogger("autonomous_perception.evaluation")
        if not self.logger.hasHandlers():
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.WARNING)
        self.reset()
        self.logger.debug("Evaluator initialized and reset.")

        self.distance_threshold = distance_threshold  # MOTA threshold

    def reset(self) -> None:
        """Reset the evaluation metrics."""
        self.logger.debug("Resetting evaluation metrics.")
        self.precisions: list[float] = []
        self.recalls: list[float] = []
        self.f1_scores: list[float] = []
        self.motas: list[float] = []
        self.motps: list[float] = []
        self.class_precisions: defaultdict = defaultdict(list)
        self.class_recalls: defaultdict = defaultdict(list)
        self.dimension_errors: list[float] = []
        self.total_objects: int = 0
        self.total_tracked: int = 0
        self.total_correct: int = 0

        # Initialize the new metric for extra tracks
        self.extra_tracks: list[int] = []
        self.extra_tracks_rates: list[float] = []  # Optional: Rate of extra tracks

    def evaluate(
        self,
        tracked_objects: list[TrackedObject],
        labels: list[GroundTruthObject],
    ) -> None:
        """Compare tracked objects with ground truth labels to compute metrics."""
        self.logger.debug(
            (
                f"Starting evaluation with {len(tracked_objects)} tracked objects and "
                f"{len(labels)} ground truth objects."
            ),
        )

        # Organize ground truth and tracked objects by class
        gt_by_class = defaultdict(list)
        for gt in labels:
            gt_by_class[gt.obj_type].append(gt)

        trk_by_class = defaultdict(list)
        for trk in tracked_objects:
            trk_by_class[trk.label].append(trk)

        frame_correct = 0  # Number of correct matches in this frame

        # Iterate over each class to compute metrics
        for cls in gt_by_class.keys():
            gts = gt_by_class[cls]
            trs = trk_by_class.get(cls, [])

            self.logger.debug(f"Evaluating class '{cls}': {len(gts)} GTs, {len(trs)} TRKs.")

            # Update total counts
            self.total_objects += len(gts)
            self.total_tracked += len(trs)
            matches = []
            # Compute cost matrix based on Euclidean distance between 3D positions
            if len(gts) > 0 and len(trs) > 0:
                cost_matrix = np.zeros((len(gts), len(trs)), dtype=np.float32)
                for i, gt in enumerate(gts):
                    for j, trk in enumerate(trs):
                        x1, y1, x2, y2 = gt.bbox
                        centroid_gt = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
                        x1, y1, x2, y2 = trk.bbox
                        centroid_trk = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
                        cost = np.linalg.norm(centroid_gt - centroid_trk)
                        cost_matrix[i, j] = cost

                # Perform Hungarian matching
                row_ind, col_ind = linear_sum_assignment(cost_matrix)

                # Determine best matches based on distance threshold,
                # and update the matches list
                for r, c in zip(row_ind, col_ind, strict=False):
                    if cost_matrix[r, c] < self.distance_threshold:
                        matches.append((r, c))

                self.logger.debug(f"Class '{cls}': {len(matches)} matches found.")

                # Update correct matches
                self.total_correct += len(matches)
                frame_correct += len(matches)  # Accumulate correct matches for this frame

                # Compute precision and recall for this class
                precision = len(matches) / len(trs) if len(trs) > 0 else 0.0
                recall = len(matches) / len(gts) if len(gts) > 0 else 0.0
                f1 = (
                    2 * precision * recall / (precision + recall)
                    if (precision + recall) > 0
                    else 0.0
                )

                self.precisions.append(precision)
                self.recalls.append(recall)
                self.f1_scores.append(f1)

                self.class_precisions[cls].append(precision)
                self.class_recalls[cls].append(recall)

                self.logger.debug(
                    f"Class '{cls}' - Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}",
                )
            else:
                # No tracked objects or no ground truth objects for this class
                precision = 0.0
                recall = 0.0
                f1 = 0.0

                self.precisions.append(precision)
                self.recalls.append(recall)
                self.f1_scores.append(f1)

                self.logger.debug(
                    f"Class '{cls}' - No matches. Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}",
                )

            # Compute dimension estimation error for matched objects
            if not matches:
                continue
            for r, c in matches:
                gt = gts[r] if r < len(gts) else None
                trk = trs[c] if c < len(trs) else None
                if gt is None or trk is None:
                    continue
                gt_dims = gt.bbox
                trk_dims = trk.bbox
                error = np.abs(np.array(gt_dims) - np.array(trk_dims))
                mean_error = np.mean(error)
                self.dimension_errors.append(mean_error)
                self.logger.debug(
                    f"Dimension error for class '{cls}', GT ID {gt.track_id}, TRK ID {trk.track_id}: {mean_error:.4f} pixels",
                )

        # Compute Extra Tracks for this frame
        extra_tracks = len(tracked_objects) - frame_correct
        extra_tracks = max(extra_tracks, 0)  # Ensure non-negative
        self.extra_tracks.append(extra_tracks)

        # Optional: Compute Extra Tracks Rate (Ratio of Extra Tracks to Ground Truth)
        ground_truth_count = len(labels)
        extra_tracks_rate = extra_tracks / ground_truth_count if ground_truth_count > 0 else 0.0
        self.extra_tracks_rates.append(extra_tracks_rate)

        self.logger.debug(f"Computed Extra Tracks for this frame: {extra_tracks}")
        self.logger.debug(f"Computed Extra Tracks Rate for this frame: {extra_tracks_rate:.4f}")

        # Compute MOTA for this frame
        false_positives = self.total_tracked - self.total_correct
        false_negatives = self.total_objects - self.total_correct
        mota = (
            1 - (false_negatives + false_positives) / self.total_objects
            if self.total_objects > 0
            else 0.0
        )
        self.motas.append(mota)
        self.logger.debug(f"Computed MOTA for this frame: {mota:.4f}")

    def report(self) -> None:
        """Aggregate and report overall evaluation metrics."""
        self.logger.debug("Generating evaluation report.")
        if not self.precisions:
            self.logger.warning("No evaluation data to report.")
            return

        # Compute average metrics
        avg_precision = sum(self.precisions) / len(self.precisions)
        avg_recall = sum(self.recalls) / len(self.recalls)
        avg_f1 = sum(self.f1_scores) / len(self.f1_scores)
        avg_mota = sum(self.motas) / len(self.motas)
        avg_dim_error = (
            sum(self.dimension_errors) / len(self.dimension_errors)
            if self.dimension_errors
            else 0.0
        )
        avg_extra_tracks = sum(self.extra_tracks) / len(self.extra_tracks)
        avg_extra_tracks_rate = (
            sum(self.extra_tracks_rates) / len(self.extra_tracks_rates)
            if self.extra_tracks_rates
            else 0.0
        )

        self.logger.info(f"Average Precision: {avg_precision:.4f}")
        self.logger.info(f"Average Recall: {avg_recall:.4f}")
        self.logger.info(f"Average F1-Score: {avg_f1:.4f}")
        self.logger.info(f"Average MOTA: {avg_mota:.4f}")
        self.logger.info(f"Average Dimension Error: {avg_dim_error:.4f} pixels")

        # Report the new Extra Tracks metrics
        self.logger.info(f"Average Extra Tracks: {avg_extra_tracks:.2f}")
        self.logger.info(f"Average Extra Tracks Rate: {avg_extra_tracks_rate:.4f}")

        # Report per-class metrics
        for cls in self.class_precisions:
            avg_cls_precision = sum(self.class_precisions[cls]) / len(self.class_precisions[cls])
            avg_cls_recall = sum(self.class_recalls[cls]) / len(self.class_recalls[cls])
            self.logger.info(
                f"Class '{cls}' - Average Precision: {avg_cls_precision:.4f}, "
                f"Average Recall: {avg_cls_recall:.4f}",
            )

    def get_average_mota(self) -> float:
        """Retrieve the average MOTA over all evaluated frames."""
        if not self.motas:
            self.logger.warning("No MOTA scores available.")
            return float("-inf")  # Assign a very low value if no data
        avg_mota = sum(self.motas) / len(self.motas)
        return avg_mota

    def get_average_precision(self) -> float:
        """Retrieve the average precision over all evaluated frames."""
        if not self.precisions:
            self.logger.warning("No precision scores available.")
            return float("-inf")
        avg_precision = sum(self.precisions) / len(self.precisions)
        return avg_precision

    # Optional: Add methods to retrieve the new metrics
    def get_average_extra_tracks(self) -> float:
        """Retrieve the average number of extra tracks over all evaluated frames."""
        if not self.extra_tracks:
            self.logger.warning("No Extra Tracks data available.")
            return 0.0
        avg_extra_tracks = sum(self.extra_tracks) / len(self.extra_tracks)
        return avg_extra_tracks

    def get_average_extra_tracks_rate(self) -> float:
        """Retrieve the average rate of extra tracks over all evaluated frames."""
        if not self.extra_tracks_rates:
            self.logger.warning("No Extra Tracks Rates available.")
            return 0.0
        avg_extra_tracks_rate = sum(self.extra_tracks_rates) / len(self.extra_tracks_rates)
        return avg_extra_tracks_rate

    def get_average_dimension_error(self) -> float:
        """Retrieve the average dimension error over all evaluated frames."""
        if not self.dimension_errors:
            self.logger.warning("No Dimension Errors available.")
            return 0.0
        avg_dim_error = sum(self.dimension_errors) / len(self.dimension_errors)
        return avg_dim_error
