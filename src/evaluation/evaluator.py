import logging
from collections import defaultdict

import numpy as np

from src.data_loader.dataset import TrackedObject
from src.evaluation.metrics import (
    compute_mota_motp,
    compute_precision_recall,
    compute_precision_recall_per_class,
)
from src.utils.label_parser import GroundTruthObject


class Evaluator:
    """Evaluate the performance of the object tracking system."""

    def __init__(self):
        """Initialize the Evaluator for computing and reporting metrics."""
        self.logger = logging.getLogger("autonomous_perception.evaluation")
        if not self.logger.hasHandlers():
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        self.reset()
        self.logger.debug("Evaluator initialized and reset.")

    def reset(self) -> None:
        """Reset the evaluation metrics."""
        self.logger.debug("Resetting evaluation metrics.")
        self.precisions = []
        self.recalls = []
        self.f1_scores = []
        self.motas = []
        self.motps = []
        self.class_precisions = defaultdict(list)
        self.class_recalls = defaultdict(list)
        self.dimension_errors = []

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

        # Compute overall precision and recall
        precision, recall = compute_precision_recall(tracked_objects, labels)
        self.logger.debug(f"Computed Precision: {precision:.4f}, Recall: {recall:.4f}")

        # Compute F1-Score
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        self.logger.debug(f"Computed F1-Score: {f1:.4f}")

        # Compute MOTA and MOTP
        mota, motp = compute_mota_motp(tracked_objects, labels)
        self.logger.debug(f"Computed MOTA: {mota:.4f}, MOTP: {motp:.4f}")

        # Compute per-class precision and recall
        class_metrics = compute_precision_recall_per_class(tracked_objects, labels)
        self.logger.debug(f"Computed per-class precision and recall: {class_metrics}")

        # Compute dimension estimation error
        dimension_error = self._compute_dimension_error(tracked_objects, labels)
        self.logger.debug(f"Computed Dimension Error: {dimension_error:.4f} meters")

        # Append metrics to respective lists
        self.precisions.append(precision)
        self.recalls.append(recall)
        self.f1_scores.append(f1)
        self.motas.append(mota)
        self.motps.append(motp)
        self.dimension_errors.append(dimension_error)

        # Append per-class metrics
        for cls, metrics in class_metrics.items():
            self.class_precisions[cls].append(metrics["precision"])
            self.class_recalls[cls].append(metrics["recall"])
            self.logger.debug(
                f"Class '{cls}' - Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}",
            )

    def _compute_dimension_error(
        self,
        tracked_objects: list[TrackedObject],
        labels: list[GroundTruthObject],
    ) -> float:
        """Compute average dimension estimation error based on closest position_3d."""
        self.logger.debug("Starting computation of dimension estimation error.")
        total_error = 0.0
        count = 0

        used_gt = set()
        MIN_THRESHOLD = 2.0  # meters

        for tr_obj in tracked_objects:
            self.logger.debug(f"Processing TrackedObject with position_3d: {tr_obj.position_3d}")
            closest_gt = None
            min_distance = float("inf")

            for gt in labels:
                if gt.track_id in used_gt:
                    self.logger.debug(
                        f"Skipping GroundTruthObject {gt.track_id} as it has already been matched."
                    )
                    continue
                distance = np.linalg.norm(np.array(tr_obj.position_3d) - np.array(gt.location))
                self.logger.debug(
                    f"Distance between TrackedObject and GroundTruthObject {gt.track_id}: {distance:.4f}",
                )
                if distance < min_distance:
                    min_distance = distance
                    closest_gt = gt

            if closest_gt and min_distance < MIN_THRESHOLD:
                self.logger.debug(
                    f"Matched TrackedObject to GroundTruthObject {closest_gt.track_id}"
                    f"with distance {min_distance:.4f}.",
                )
                gt_dims = closest_gt.dimensions  # [h, w, l]
                tr_dims = tr_obj.dimensions
                self.logger.debug(
                    f"GroundTruth dimensions: {gt_dims}, Tracked dimensions: {tr_dims}",
                )

                # Compute absolute difference
                error = np.abs(np.array(gt_dims) - np.array(tr_dims))
                self.logger.debug(f"Dimension error: {error}")

                # Average the error across dimensions
                mean_error = np.mean(error)
                self.logger.debug(
                    f"Mean dimension error: {mean_error:.4f} meters",
                )

                total_error += mean_error
                count += 1
                used_gt.add(closest_gt.track_id)
            else:
                self.logger.debug(
                    f"No matching GroundTruthObject found within threshold for TrackedObject"
                    f"at position {tr_obj.position_3d}.",
                )

        average_error = float(total_error / count) if count > 0 else 0.0
        self.logger.debug(
            f"Average Dimension Error: {average_error:.4f} meters over {count} matched objects.",
        )
        return average_error

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
        avg_motp = sum(self.motps) / len(self.motps)
        avg_dim_error = sum(self.dimension_errors) / len(self.dimension_errors)

        self.logger.info(f"Average Precision: {avg_precision:.4f}")
        self.logger.info(f"Average Recall: {avg_recall:.4f}")
        self.logger.info(f"Average F1-Score: {avg_f1:.4f}")
        self.logger.info(f"Average MOTA: {avg_mota:.4f}")
        self.logger.info(f"Average MOTP: {avg_motp:.4f}")
        self.logger.info(f"Average Dimension Error: {avg_dim_error:.4f} meters")

        # Report per-class metrics
        for cls in self.class_precisions:
            avg_cls_precision = sum(self.class_precisions[cls]) / len(self.class_precisions[cls])
            avg_cls_recall = sum(self.class_recalls[cls]) / len(self.class_recalls[cls])
            self.logger.info(
                f"Class '{cls}' - Average Precision: {avg_cls_precision:.4f}, Average Recall: {avg_cls_recall:.4f}",
            )
