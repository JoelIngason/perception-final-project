def compute_precision_recall(tracked_objects: list, labels: list[dict]) -> tuple[float, float]:
    """Compute precision and recall metrics based on tracked objects and ground truth labels."""
    TP = 0
    FP = 0
    FN = 0

    matched_gt = set()
    matched_tr = set()

    for tr_idx, tr_obj in enumerate(tracked_objects):
        tr_bbox = tr_obj.bbox
        tr_label = tr_obj.label.lower()
        tr_iou_max = 0
        tr_matched_gt_idx = -1

        for gt_idx, gt in enumerate(labels):
            if gt_idx in matched_gt:
                continue
            gt_bbox = gt["bbox"]
            gt_label = gt["type"].lower()
            if tr_label != gt_label:
                continue
            iou = compute_iou(tr_bbox, gt_bbox)
            if iou > tr_iou_max:
                tr_iou_max = iou
                tr_matched_gt_idx = gt_idx

        if tr_iou_max >= 0.5:
            TP += 1
            matched_gt.add(tr_matched_gt_idx)
            matched_tr.add(tr_idx)
        else:
            FP += 1

    FN = len(labels) - len(matched_gt)

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0

    return precision, recall


def compute_iou(boxA: list[float], boxB: list[float]) -> float:
    """Compute the Intersection over Union (IoU) of two bounding boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interWidth = max(0, xB - xA + 1)
    interHeight = max(0, yB - yA + 1)
    interArea = interWidth * interHeight

    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    iou = (
        interArea / float(boxAArea + boxBArea - interArea)
        if (boxAArea + boxBArea - interArea) > 0
        else 0.0
    )
    return iou
