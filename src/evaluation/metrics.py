from collections import defaultdict

from src.data_loader.dataset import TrackedObject

# Constants
IOU_THRESHOLD = 0.5


class Metrics:
    TP: int
    FP: int
    matched_gt: set[int]

    def __init__(self):
        self.TP = 0
        self.FP = 0
        self.matched_gt = set()


def compute_iou(boxA: list[float | int], boxB: list[float | int]) -> float:
    """Compute the Intersection over Union (IoU) of two bounding boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interWidth = max(0, xB - xA)
    interHeight = max(0, yB - yA)
    interArea = interWidth * interHeight

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    unionArea = boxAArea + boxBArea - interArea
    return interArea / unionArea if unionArea > 0 else 0.0


def compute_precision_recall(
    tracked_objects: list[TrackedObject],
    labels: list[dict],
) -> tuple[float, float]:
    """Compute precision and recall metrics based on tracked objects and ground truth labels."""
    TP = 0
    FP = 0
    matched_gt = set()

    for tr_obj in tracked_objects:
        tr_bbox = tr_obj.bbox  # [x, y, w, h]
        tr_label = tr_obj.to_supported_label().lower()
        # Convert to [x1, y1, x2, y2]
        tr_bbox_xyxy = [
            tr_bbox[0],
            tr_bbox[1],
            tr_bbox[0] + tr_bbox[2],
            tr_bbox[1] + tr_bbox[3],
        ]

        tr_iou_max = 0.0
        tr_matched_gt_idx = -1

        for gt_idx, gt in enumerate(labels):
            if gt_idx in matched_gt:
                continue
            gt_bbox = gt["bbox"]  # [x1, y1, x2, y2]
            gt_label = gt["type"].lower()
            if tr_label != gt_label:
                continue
            iou = compute_iou(tr_bbox_xyxy, gt_bbox)  # type: ignore
            if iou > tr_iou_max:
                tr_iou_max = iou
                tr_matched_gt_idx = gt_idx

        if tr_iou_max >= IOU_THRESHOLD:
            TP += 1
            matched_gt.add(tr_matched_gt_idx)
        else:
            FP += 1

    FN = len(labels) - len(matched_gt)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0

    return precision, recall


def compute_mota_motp(
    tracked_objects: list[TrackedObject],
    labels: list[dict],
    iou_threshold: float = IOU_THRESHOLD,
) -> tuple[float, float]:
    """Compute Multiple Object Tracking Accuracy (MOTA) and Precision (MOTP)."""
    TP = 0
    FP = 0
    FN = 0
    total_iou = 0.0
    matched_gt = set()

    for tr_obj in tracked_objects:
        tr_bbox = tr_obj.bbox  # [x, y, w, h]
        tr_label = tr_obj.to_supported_label().lower()
        tr_bbox_xyxy = [
            tr_bbox[0],
            tr_bbox[1],
            tr_bbox[0] + tr_bbox[2],
            tr_bbox[1] + tr_bbox[3],
        ]

        tr_iou_max = 0.0
        tr_matched_gt_idx = -1

        for gt_idx, gt in enumerate(labels):
            if gt_idx in matched_gt:
                continue
            gt_bbox = gt["bbox"]  # [x1, y1, x2, y2]
            gt_label = gt["type"].lower()
            if tr_label != gt_label:
                continue
            iou = compute_iou(tr_bbox_xyxy, gt_bbox)  # type: ignore
            if iou > tr_iou_max:
                tr_iou_max = iou
                tr_matched_gt_idx = gt_idx

        if tr_iou_max >= iou_threshold:
            TP += 1
            matched_gt.add(tr_matched_gt_idx)
            total_iou += tr_iou_max
        else:
            FP += 1

    FN = len(labels) - len(matched_gt)
    mota = 1 - (FP + FN) / len(labels) if len(labels) > 0 else 0.0
    motp = total_iou / TP if TP > 0 else 0.0

    return mota, motp


def compute_precision_recall_per_class(
    tracked_objects: list[TrackedObject],
    labels: list[dict],
) -> dict[str, dict[str, float]]:
    """Compute precision and recall metrics for each object class."""
    class_metrics: dict[str, Metrics] = defaultdict(Metrics)

    for tr_obj in tracked_objects:
        tr_bbox = tr_obj.bbox  # [x, y, w, h]
        tr_label = tr_obj.to_supported_label().lower()
        tr_bbox_xyxy = [
            tr_bbox[0],
            tr_bbox[1],
            tr_bbox[0] + tr_bbox[2],
            tr_bbox[1] + tr_bbox[3],
        ]

        matched = False
        for gt_idx, gt in enumerate(labels):
            if gt_idx in class_metrics[tr_label].matched_gt:
                continue
            gt_bbox = gt["bbox"]  # [x1, y1, x2, y2]
            gt_label = gt["type"].lower()
            if tr_label != gt_label:
                continue
            iou = compute_iou(tr_bbox_xyxy, gt_bbox)  # type: ignore
            if iou >= IOU_THRESHOLD:
                class_metrics[tr_label].TP += 1
                class_metrics[tr_label].matched_gt.add(gt_idx)
                matched = True
                break

        if not matched:
            class_metrics[tr_label].FP += 1

    precision_recall = {}
    for cls, metrics in class_metrics.items():
        TP = metrics.TP
        FP = metrics.FP
        FN = sum(1 for gt in labels if gt["type"].lower() == cls) - len(metrics.matched_gt)
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        precision_recall[cls] = {"precision": precision, "recall": recall}

    return precision_recall
