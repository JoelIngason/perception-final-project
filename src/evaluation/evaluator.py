import logging

from src.data_loader.dataset import TrackedObject
from src.evaluation.metrics import compute_precision_recall


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
        self.precisions = []
        self.recalls = []
        self.f1_scores = []

    def evaluate(self, tracked_objects: list[TrackedObject], labels: list[dict]) -> None:
        """Compare tracked objects with ground truth labels to compute metrics."""
        precision, recall = compute_precision_recall(tracked_objects, labels)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        self.precisions.append(precision)
        self.recalls.append(recall)
        self.f1_scores.append(f1)
        self.logger.debug(
            f"Evaluation metrics - Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}",
        )

    def report(self) -> None:
        """Aggregate and report overall evaluation metrics."""
        if not self.precisions or not self.recalls:
            self.logger.warning("No evaluation data to report.")
            return
        avg_precision = sum(self.precisions) / len(self.precisions)
        avg_recall = sum(self.recalls) / len(self.recalls)
        avg_f1 = sum(self.f1_scores) / len(self.f1_scores)
        self.logger.info(f"Average Precision: {avg_precision:.4f}")
        self.logger.info(f"Average Recall: {avg_recall:.4f}")
        self.logger.info(f"Average F1-Score: {avg_f1:.4f}")
