import logging
from collections import defaultdict

import numpy as np

from src.data_loader.dataset import TrackedObject
from src.evaluation.metrics import (
    compute_mota_motp,
    compute_precision_recall,
    compute_precision_recall_per_class,
)


class Evaluator:
    """Evaluate the performance of the object tracking system."""

    def __init__(self):
        """Initialize the Evaluator for computing and reporting metrics."""
        self.logger = logging.getLogger("autonomous_perception.evaluation")
        self.logger.setLevel(logging.INFO)
        if not self.logger.hasHandlers():
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        self.reset()

    def reset(self) -> None:
        """Reset the evaluation metrics."""
        self.precisions = []
        self.recalls = []
        self.f1_scores = []
        self.motas = []
        self.motps = []
        self.class_precisions = defaultdict(list)
        self.class_recalls = defaultdict(list)
        self.dimension_errors = []

    def evaluate(self, tracked_objects: list[TrackedObject], labels: list[dict]) -> None:
        """Compare tracked objects with ground truth labels to compute metrics."""
        precision, recall = compute_precision_recall(tracked_objects, labels)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        mota, motp = compute_mota_motp(tracked_objects, labels)
        class_metrics = compute_precision_recall_per_class(tracked_objects, labels)
        dimension_error = self._compute_dimension_error(tracked_objects, labels)

        self.precisions.append(precision)
        self.recalls.append(recall)
        self.f1_scores.append(f1)
        self.motas.append(mota)
        self.motps.append(motp)
        self.dimension_errors.append(dimension_error)

        for cls, metrics in class_metrics.items():
            self.class_precisions[cls].append(metrics["precision"])
            self.class_recalls[cls].append(metrics["recall"])

    def _compute_dimension_error(self, tracked_objects, labels):
        """Compute average dimension estimation error."""
        total_error = 0.0
        count = 0
        for tr_obj in tracked_objects:
            for gt in labels:
                if tr_obj.track_id == gt.get("track_id"):
                    gt_dims = gt["dimensions"]  # [h, w, l]
                    tr_dims = tr_obj.dimensions
                    error = np.abs(np.array(gt_dims) - np.array(tr_dims))
                    total_error += np.mean(error)
                    count += 1
                    break
        return total_error / count if count > 0 else 0.0

    def report(self) -> None:
        """Aggregate and report overall evaluation metrics."""
        if not self.precisions:
            self.logger.warning("No evaluation data to report.")
            return

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

        # Per-class metrics
        for cls in self.class_precisions:
            avg_cls_precision = sum(self.class_precisions[cls]) / len(self.class_precisions[cls])
            avg_cls_recall = sum(self.class_recalls[cls]) / len(self.class_recalls[cls])
            self.logger.info(
                f"Class '{cls}' - Precision: {avg_cls_precision:.4f}, Recall: {avg_cls_recall:.4f}",
            )
