import logging

from src.evaluation.metrics import compute_precision_recall


class Evaluator:
    def __init__(self):
        self.logger = logging.getLogger('autonomous_perception.evaluation')
        self.predictions = []
        self.ground_truth = []

    def evaluate(self, tracked_objects, labels):
        # Compare tracked_objects with labels to compute metrics
        # Placeholder for evaluation logic
        precision, recall = compute_precision_recall(tracked_objects, labels)
        self.predictions.append(precision)
        self.ground_truth.append(recall)
        self.logger.debug(f"Evaluation metrics - Precision: {precision}, Recall: {recall}")

    def report(self):
        # Aggregate and report overall metrics
        avg_precision = sum(self.predictions) / len(self.predictions) if self.predictions else 0
        avg_recall = sum(self.ground_truth) / len(self.ground_truth) if self.ground_truth else 0
        self.logger.info(f"Average Precision: {avg_precision}")
        self.logger.info(f"Average Recall: {avg_recall}")
