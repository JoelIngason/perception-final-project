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

        self.predictions: list[float] = []
        self.ground_truth: list[float] = []

    def evaluate(self, tracked_objects: list[TrackedObject], labels: list[dict]) -> None:
        """
        Compare tracked objects with ground truth labels to compute metrics.

        Args:
            tracked_objects (List[TrackedObject]): List of tracked objects.
            labels (List[Dict]): Ground truth labels for the current frame.

        """
        precision, recall = compute_precision_recall(tracked_objects, labels)
        self.predictions.append(precision)
        self.ground_truth.append(recall)
        self.logger.debug(f"Evaluation metrics - Precision: {precision}, Recall: {recall}")

    def report(self) -> None:
        """Aggregate and report overall evaluation metrics."""
        if not self.predictions or not self.ground_truth:
            self.logger.warning("No evaluation data to report.")
            return

        avg_precision = sum(self.predictions) / len(self.predictions)
        avg_recall = sum(self.ground_truth) / len(self.ground_truth)
        self.logger.info(f"Average Precision: {avg_precision:.4f}")
        self.logger.info(f"Average Recall: {avg_recall:.4f}")
