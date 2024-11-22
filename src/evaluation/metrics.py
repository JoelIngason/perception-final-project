from collections import defaultdict

import numpy as np
from scipy.optimize import linear_sum_assignment

from src.data_loader.dataset import TrackedObject
from src.utils.label_parser import GroundTruthObject

# Constants
DISTANCE_THRESHOLD = 2.0  # meters


class Metrics:
    TP: int
    FP: int
    FN: int
    matched_gt: set[int]

    def __init__(self):
        self.TP = 0
        self.FP = 0
        self.FN = 0
        self.matched_gt = set()


def compute_euclidean_distance(
    pos1: list[float],
    pos2: list[float],
) -> float:
    """Compute the Euclidean distance between two 3D positions."""
    return float(np.linalg.norm(np.array(pos1) - np.array(pos2)))


def compute_precision_recall(
    tracked_objects: list[TrackedObject],
    ground_truths: list[GroundTruthObject],
    distance_threshold: float = DISTANCE_THRESHOLD,
) -> tuple[float, float]:
    """
    Compute precision and recall metrics based on tracked objects and ground truth labels using 3D positions.

    Args:
        tracked_objects (List[TrackedObject]): List of tracked objects.
        ground_truths (List[GroundTruthObject]): List of ground truth objects.
        distance_threshold (float): Maximum distance (in meters) to consider a match.

    Returns:
        Tuple[float, float]: Precision and Recall values.

    """
    if not tracked_objects:
        precision = 0.0
        recall = 0.0 if ground_truths else 1.0
        return precision, recall

    if not ground_truths:
        precision = 0.0
        recall = 1.0
        return precision, recall

    # Number of tracks and ground truths
    num_tracks = len(tracked_objects)
    num_gt = len(ground_truths)

    # Initialize distance matrix
    distance_matrix = np.zeros((num_tracks, num_gt), dtype=np.float32)

    for i, tr_obj in enumerate(tracked_objects):
        for j, gt in enumerate(ground_truths):
            if tr_obj.label.lower() != gt.obj_type.lower():
                # Assign a large distance if labels do not match
                distance_matrix[i, j] = distance_threshold + 1.0
            elif tr_obj.position_3d is None or gt.location is None:
                # Assign a large distance if position is missing
                distance_matrix[i, j] = distance_threshold + 1.0
            else:
                distance = compute_euclidean_distance(tr_obj.position_3d, gt.location)
                distance_matrix[i, j] = distance

    # Apply Hungarian algorithm for optimal assignment
    row_indices, col_indices = linear_sum_assignment(distance_matrix)

    # Initialize metrics
    TP = 0
    FP = 0
    matched_gt = set()

    for row, col in zip(row_indices, col_indices, strict=False):
        if distance_matrix[row, col] <= distance_threshold:
            TP += 1
            matched_gt.add(col)
        else:
            FP += 1

    # False Positives: Tracked objects not matched
    FP += num_tracks - TP

    # False Negatives: Ground truths not matched
    FN = num_gt - TP

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0

    return precision, recall


def compute_mota_motp(
    tracked_objects: list[TrackedObject],
    ground_truths: list[GroundTruthObject],
    distance_threshold: float = DISTANCE_THRESHOLD,
) -> tuple[float, float]:
    """
    Compute Multiple Object Tracking Accuracy (MOTA) and Precision (MOTP) using 3D positions.

    Args:
        tracked_objects (List[TrackedObject]): List of tracked objects.
        ground_truths (List[GroundTruthObject]): List of ground truth objects.
        distance_threshold (float): Maximum distance (in meters) to consider a match.

    Returns:
        Tuple[float, float]: MOTA and MOTP values.

    """
    if not ground_truths:
        return 1.0, 0.0 if not tracked_objects else 0.0

    # Number of tracks and ground truths
    num_tracks = len(tracked_objects)
    num_gt = len(ground_truths)

    # Initialize distance matrix
    distance_matrix = np.zeros((num_tracks, num_gt), dtype=np.float32)

    for i, tr_obj in enumerate(tracked_objects):
        for j, gt in enumerate(ground_truths):
            if (
                tr_obj.label.lower() != gt.obj_type.lower()
                or tr_obj.position_3d is None
                or gt.location is None
            ):
                distance_matrix[i, j] = distance_threshold + 1.0
            else:
                distance = compute_euclidean_distance(tr_obj.position_3d, gt.location)
                distance_matrix[i, j] = distance

    # Apply Hungarian algorithm for optimal assignment
    row_indices, col_indices = linear_sum_assignment(distance_matrix)

    # Initialize metrics
    TP = 0
    FP = 0
    total_iou = 0.0  # Not applicable for 3D, consider average distance instead
    matched_gt = set()

    for row, col in zip(row_indices, col_indices, strict=False):
        if distance_matrix[row, col] <= distance_threshold:
            TP += 1
            matched_gt.add(col)
            # For MOTP, sum the inverse of distance or another metric
            # Here, we'll use the actual distance
            total_iou += distance_matrix[row, col]
        else:
            FP += 1

    # False Positives: Tracked objects not matched
    FP += num_tracks - TP

    # False Negatives: Ground truths not matched
    FN = num_gt - TP

    # Compute MOTA
    mota = 1 - (FP + FN) / num_gt if num_gt > 0 else 1.0

    # Compute MOTP
    motp = total_iou / TP if TP > 0 else 0.0

    return float(mota), float(motp)


def compute_precision_recall_per_class(
    tracked_objects: list[TrackedObject],
    ground_truths: list[GroundTruthObject],
    distance_threshold: float = DISTANCE_THRESHOLD,
) -> dict[str, dict[str, float]]:
    """
    Compute precision and recall metrics for each object class based on 3D positions.

    Args:
        tracked_objects (List[TrackedObject]): List of tracked objects.
        ground_truths (List[GroundTruthObject]): List of ground truth objects.
        distance_threshold (float): Maximum distance (in meters) to consider a match.

    Returns:
        Dict[str, Dict[str, float]]: Dictionary with classes as keys and precision & recall as values.

    """
    class_metrics: dict[str, Metrics] = defaultdict(Metrics)

    # Create a list of ground truths per class
    gt_per_class: dict[str, list[GroundTruthObject]] = defaultdict(list)
    for gt in ground_truths:
        cls = gt.obj_type.lower()
        gt_per_class[cls].append(gt)

    # Create a list of tracked objects per class
    tr_per_class: dict[str, list[TrackedObject]] = defaultdict(list)
    for tr_obj in tracked_objects:
        cls = tr_obj.to_supported_label().lower()
        tr_per_class[cls].append(tr_obj)

    for cls, tr_objs in tr_per_class.items():
        gt_objs = gt_per_class.get(cls, [])
        if not gt_objs:
            # All tracked objects are False Positives
            class_metrics[cls].FP += len(tr_objs)
            continue

        num_tr = len(tr_objs)
        num_gt = len(gt_objs)

        # Initialize distance matrix
        distance_matrix = np.zeros((num_tr, num_gt), dtype=np.float32)

        for i, tr_obj in enumerate(tr_objs):
            for j, gt in enumerate(gt_objs):
                if tr_obj.position_3d is None or gt.location is None:
                    distance_matrix[i, j] = distance_threshold + 1.0
                else:
                    distance = compute_euclidean_distance(tr_obj.position_3d, gt.location)
                    distance_matrix[i, j] = distance

        # Apply Hungarian algorithm
        row_indices, col_indices = linear_sum_assignment(distance_matrix)

        for row, col in zip(row_indices, col_indices, strict=False):
            if distance_matrix[row, col] <= distance_threshold:
                class_metrics[cls].TP += 1
                class_metrics[cls].matched_gt.add(col)
            else:
                class_metrics[cls].FP += 1

        # False Positives: Tracked objects not matched
        class_metrics[cls].FP += num_tr - class_metrics[cls].TP

        # False Negatives: Ground truths not matched
        class_metrics[cls].FN += num_gt - class_metrics[cls].TP

    # Compute precision and recall per class
    precision_recall = {}
    for cls, metrics in class_metrics.items():
        TP = metrics.TP
        FP = metrics.FP
        FN = metrics.FN
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        precision_recall[cls] = {"precision": precision, "recall": recall}

    return precision_recall
