import argparse
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import yaml
from sklearn.metrics import average_precision_score
from ultralytics import YOLO

sys.path.append("")
from src.utils.label_parser import GroundTruthObject, parse_labels


def load_model(model_path: str = "yolov8/best.pt") -> YOLO:
    """
    Load a pretrained YOLO model.

    Args:
        model_path (str, optional): Path to the YOLO weights file. Defaults to "yolov8/best.pt".

    Returns:
        YOLO: The loaded YOLO model instance.

    Raises:
        FileNotFoundError: If the model file does not exist.

    """
    logger = logging.getLogger("autonomous_perception.models")
    if not Path(model_path).is_file():
        logger.error(f"Model file not found at {model_path}")
        raise FileNotFoundError(f"Model file not found at {model_path}")
    logger.info(f"Loading YOLO model from {model_path}")
    return YOLO(model_path, verbose=False)


def get_model(model_path: str | None = None) -> YOLO:
    """
    Get the YOLO model instance.

    Args:
        model_path (Optional[str], optional): Path to the YOLO weights file. If None, uses default path. Defaults to None.

    Returns:
        YOLO: The YOLO model instance.

    """
    if model_path is None:
        model_path = str(Path(__file__).parent / "yolov8/best.pt")
    return load_model(str(model_path))


def compute_iou(box1: tuple[int, int, int, int], box2: tuple[int, int, int, int]) -> float:
    """
    Compute the Intersection over Union (IoU) of two bounding boxes.

    Args:
        box1 (Tuple[int, int, int, int]): Bounding box in the format (x1, y1, x2, y2).
        box2 (Tuple[int, int, int, int]): Bounding box in the format (x1, y1, x2, y2).

    Returns:
        float: IoU value.

    """
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    # Calculate intersection
    xi1 = max(x1_1, x1_2)
    yi1 = max(y1_1, y1_2)
    xi2 = min(x2_1, x2_2)
    yi2 = min(y2_1, y2_2)
    inter_width = max(0, xi2 - xi1 + 1)
    inter_height = max(0, yi2 - yi1 + 1)
    inter_area = inter_width * inter_height

    # Calculate union
    box1_area = (x2_1 - x1_1 + 1) * (y2_1 - y1_1 + 1)
    box2_area = (x2_2 - x1_2 + 1) * (y2_2 - y1_2 + 1)
    union_area = box1_area + box2_area - inter_area

    if union_area == 0:
        return 0.0

    iou = inter_area / union_area
    return iou


def match_detections_to_ground_truth(
    detections: list[dict],
    ground_truths: list[GroundTruthObject],
    iou_threshold: float = 0.5,
) -> tuple[int, int, int, list[int]]:
    """
    Match detected bounding boxes to ground truth boxes and compute TP, FP, FN.

    Args:
        detections (List[Dict]): List of detections with keys 'bbox' and 'confidence'.
        ground_truths (List[GroundTruthObject]): List of ground truth objects.
        iou_threshold (float, optional): IoU threshold to consider a detection as True Positive. Defaults to 0.5.

    Returns:
        Tuple[int, int, int, List[int]]: Number of True Positives, False Positives, False Negatives, and list of matched ground truth IDs.

    """
    tp = 0
    fp = 0
    fn = 0

    matched_gt_ids = set()

    for det in detections:
        det_bbox = det["bbox"]
        det_conf = det["confidence"]
        best_iou = 0.0
        best_gt_idx = -1
        for idx, gt in enumerate(ground_truths):
            if gt.track_id in matched_gt_ids:
                continue
            gt_bbox = gt.bbox  # [x1, y1, x2, y2]
            iou = compute_iou(det_bbox, gt_bbox)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = idx
        if best_iou >= iou_threshold and best_gt_idx != -1:
            tp += 1
            matched_gt_ids.add(ground_truths[best_gt_idx].track_id)
        else:
            fp += 1

    fn = len(ground_truths) - len(matched_gt_ids)

    return tp, fp, fn, list(matched_gt_ids)


def evaluate_yolo_performance(
    model: YOLO,
    image_paths: list[str],
    labels: dict[int, list[GroundTruthObject]],
    iou_threshold: float = 0.5,
    score_threshold: float = 0.5,
    show: bool = False,
    save_plots: bool = False,
    save_dir: str | None = None,
    seq: str = "seq_01",
) -> None:
    """
    Evaluate YOLO model performance against ground truth labels, including mAP.

    Args:
        model (YOLO): Loaded YOLO model.
        image_paths (List[str]): List of image file paths.
        labels (Dict[int, List[GroundTruthObject]]): Ground truth labels per frame.
        iou_threshold (float, optional): IoU threshold for matching detections to ground truth. Defaults to 0.5.
        score_threshold (float, optional): Confidence score threshold for detections. Defaults to 0.5.
        show (bool, optional): Whether to display images with detections. Defaults to False.
        save_plots (bool, optional): Whether to save the generated plots. Defaults to False.
        save_dir (str, optional): Directory to save the plots. Required if save_plots=True.

    """
    logger = logging.getLogger("YOLO_Evaluation")
    logger.info("Starting YOLO model performance evaluation...")

    if save_plots and save_dir is None:
        logger.error("save_dir must be specified if save_plots is True.")
        raise ValueError("save_dir must be specified if save_plots is True.")

    if save_plots:
        os.makedirs(save_dir, exist_ok=True)
        logger.info(f"Saving plots to directory: {save_dir}")

    total_tp = 0
    total_fp = 0
    total_fn = 0

    # Lists to store per-frame metrics
    per_frame_precision = []
    per_frame_recall = []
    per_frame_f1 = []
    per_frame_mota = []

    # For mAP calculation
    classes = set()
    gt_per_class = defaultdict(list)  # class_name -> list of ground truth boxes
    det_per_class = defaultdict(list)  # class_name -> list of detections

    # Mapping from class IDs to class names (assuming YOLO class IDs correspond to labels)
    # This mapping needs to be defined based on your YOLO model's classes.
    # For example:
    # class_id_to_name = {0: 'Pedestrian', 1: 'Cyclist', 2: 'Car'}
    # Adjust this mapping as per your model.
    class_id_to_name = {
        0: "Pedestrian",
        1: "Cyclist",
        2: "Car",
        # Add other classes as needed
    }

    for frame_idx, image_path in enumerate(image_paths):
        img = cv2.imread(image_path)
        if img is None:
            logger.error(f"Could not read image at {image_path}")
            # Append NaNs for metrics
            per_frame_precision.append(np.nan)
            per_frame_recall.append(np.nan)
            per_frame_f1.append(np.nan)
            per_frame_mota.append(np.nan)
            continue

        # Run YOLO model inference
        results = model(img)

        detections = []
        for result in results:
            for box in result.boxes:
                conf = box.conf.item()
                cls = int(box.cls.item())
                if conf < score_threshold:
                    continue
                # Assuming YOLO returns boxes in (x1, y1, x2, y2) format
                bbox = box.xyxy.tolist()[0]  # [x1, y1, x2, y2]
                bbox = [int(coord) for coord in bbox]
                class_name = class_id_to_name.get(cls, f"class_{cls}")
                detections.append({"bbox": bbox, "confidence": conf, "class": class_name})

        # Get ground truth for this frame
        gt_objects = labels.get(frame_idx, [])
        # Take out occluded objects
        # gt_objects = [gt for gt in gt_objects if gt.occluded not in (2, 3)]
        # for gt in gt_objects:
        #    if gt.occluded == 2:
        #        print("Occluded object")

        # contains left, top, right, bottom pixel coordinates
        mike_location = [660, 350, 910, 100]  # [x1, y1, x2, y2]
        # remove gt_objects that are within the mike_location box
        if seq == "seq_02":
            gt_objects = [
                gt
                for gt in gt_objects
                if (gt.bbox[0] < mike_location[0] and gt.bbox[1] < mike_location[1])
                or (gt.bbox[2] > mike_location[2] and gt.bbox[3] > mike_location[3])
            ]
        # Update class set and ground truths per class
        for gt in gt_objects:
            classes.add(gt.obj_type)
            gt_per_class[gt.obj_type].append(
                {
                    "bbox": gt.bbox,  # [x1, y1, x2, y2]
                    "image_id": frame_idx,
                    "matched": False,  # To keep track during matching
                },
            )

        # Update detections per class
        for det in detections:
            det_per_class[det["class"]].append(
                {"bbox": det["bbox"], "score": det["confidence"], "image_id": frame_idx},
            )

        # Match detections to ground truth
        tp, fp, fn, matched_gt_ids = match_detections_to_ground_truth(
            detections,
            gt_objects,
            iou_threshold=iou_threshold,
        )
        total_tp += tp
        total_fp += fp
        total_fn += fn

        # Calculate per-frame metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = (
            2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        )
        mota = 1 - (fp + fn) / len(gt_objects) if len(gt_objects) > 0 else np.nan

        per_frame_precision.append(precision)
        per_frame_recall.append(recall)
        per_frame_f1.append(f1_score)
        per_frame_mota.append(mota)

        logger.debug(
            f"Seq {seq} Frame {frame_idx}: TP={tp}, FP={fp}, FN={fn}, Precision={precision:.4f}, Recall={recall:.4f}, F1={f1_score:.4f}, MOTA={mota:.4f}",
        )

        if show or save_plots:
            # Create a copy of the image to draw on
            img_display = img.copy()

            # Draw ground truth boxes in green
            for gt in gt_objects:
                x1, y1, x2, y2 = map(int, gt.bbox)
                cv2.rectangle(img_display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    img_display,
                    f"GT {gt.obj_type}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                )
            # Draw detection boxes in red
            for det in detections:
                x1, y1, x2, y2 = det["bbox"]
                cls_name = det["class"]
                conf = det["confidence"]
                cv2.rectangle(img_display, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(
                    img_display,
                    f"Det {cls_name} {conf:.2f}",
                    (x1, y2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    1,
                )

            if show:
                cv2.imshow(f"Frame {frame_idx} - YOLO Detection", img_display)
                cv2.waitKey(1)

            if save_plots:
                # Save the image with detections
                frame_filename = f"{seq}-frame_{frame_idx:04d}.png"
                save_path = os.path.join(save_dir, frame_filename)
                cv2.imwrite(save_path, img_display)
                logger.debug(f"Saved detection visualization to {save_path}")

    # After processing all frames, compute per-class AP and mAP
    # Prepare data for AP calculation
    ap_per_class = {}
    for cls in classes:
        detections_cls = det_per_class[cls]
        gt_cls = gt_per_class[cls]
        npos = len(gt_cls)

        if npos == 0:
            logger.warning(f"No ground truth for class '{cls}'. Skipping AP calculation.")
            ap_per_class[cls] = np.nan
            continue

        # Sort detections by descending confidence
        detections_cls = sorted(detections_cls, key=lambda x: x["score"], reverse=True)

        y_scores = []
        y_true = []

        # Create a list to keep track of which ground truths have been matched
        gt_matched = defaultdict(lambda: np.zeros(len(gt_cls)))

        for det in detections_cls:
            image_id = det["image_id"]
            det_bbox = det["bbox"]
            score = det["score"]
            y_scores.append(score)

            # Initialize as False
            matched = False
            max_iou = 0.0
            max_gt_idx = -1

            for gt_idx, gt in enumerate(gt_cls):
                if gt["image_id"] != image_id:
                    continue
                if gt_matched[image_id][gt_idx]:
                    continue
                iou = compute_iou(det_bbox, gt["bbox"])
                if iou > max_iou:
                    max_iou = iou
                    max_gt_idx = gt_idx

            if max_iou >= iou_threshold and not gt_matched[image_id][max_gt_idx]:
                y_true.append(1)  # True Positive
                gt_matched[image_id][max_gt_idx] = 1
                matched = True
            if not matched:
                y_true.append(0)  # False Positive

        # Compute AP using sklearn
        if len(y_true) == 0:
            ap = 0.0
            logger.warning(f"No detections for class '{cls}'. AP set to 0.")
        else:
            ap = average_precision_score(y_true, y_scores)
        ap_per_class[cls] = ap
        logger.info(f"Class '{cls}': AP={ap:.4f}")

    # Compute mAP
    valid_aps = [ap for ap in ap_per_class.values() if not np.isnan(ap)]
    mAP = np.mean(valid_aps) if valid_aps else np.nan
    logger.info(f"Mean Average Precision (mAP): {mAP:.4f}")

    if save_plots:
        # Save per-class APs to a text file
        ap_text_path = os.path.join(save_dir, f"{seq}_per_class_AP.txt")
        with open(ap_text_path, "w") as f:
            f.write("Per-Class Average Precision (AP):\n")
            for cls, ap in ap_per_class.items():
                f.write(f"Class '{cls}': AP={ap:.4f}\n")
            f.write(f"\nMean Average Precision (mAP): {mAP:.4f}\n")
        logger.info(f"Saved per-class AP and mAP to {ap_text_path}")

        # Plot per-class AP
        classes_sorted = sorted(ap_per_class.keys())
        ap_values = [ap_per_class[cls] for cls in classes_sorted]

        plt.figure(figsize=(12, 8))
        plt.bar(classes_sorted, ap_values, color="skyblue")
        plt.xlabel("Classes")
        plt.ylabel("Average Precision (AP)")
        plt.title("Per-Class Average Precision (AP)")
        plt.ylim(0, 1)
        for idx, ap in enumerate(ap_values):
            plt.text(idx, ap + 0.01, f"{ap:.2f}", ha="center", va="bottom")
        plt.tight_layout()
        per_class_ap_plot_path = os.path.join(save_dir, f"{seq}_per_class_AP.png")
        plt.savefig(per_class_ap_plot_path)
        logger.info(f"Saved per-class AP plot to {per_class_ap_plot_path}")
        plt.close()

        # Plot mAP
        plt.figure(figsize=(6, 6))
        plt.bar(["mAP"], [mAP], color="orange")
        plt.ylim(0, 1)
        plt.ylabel("Mean Average Precision (mAP)")
        plt.title("Mean Average Precision (mAP)")
        plt.text(0, mAP + 0.01, f"{mAP:.2f}", ha="center", va="bottom")
        plt.tight_layout()
        mAP_plot_path = os.path.join(save_dir, f"{seq}_mAP.png")
        plt.savefig(mAP_plot_path)
        logger.info(f"Saved mAP plot to {mAP_plot_path}")
        plt.close()

    # Calculate overall metrics
    overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    overall_f1 = (
        2 * overall_precision * overall_recall / (overall_precision + overall_recall)
        if (overall_precision + overall_recall) > 0
        else 0.0
    )
    overall_mota = (
        1 - (total_fp + total_fn) / sum(len(gt) for gt in labels.values())
        if sum(len(gt) for gt in labels.values()) > 0
        else np.nan
    )

    logger.info("YOLO Evaluation Results:")
    logger.info(f"Total True Positives (TP): {total_tp}")
    logger.info(f"Total False Positives (FP): {total_fp}")
    logger.info(f"Total False Negatives (FN): {total_fn}")
    logger.info(f"Overall Precision: {overall_precision:.4f}")
    logger.info(f"Overall Recall: {overall_recall:.4f}")
    logger.info(f"Overall F1 Score: {overall_f1:.4f}")
    logger.info(f"Overall MOTA: {overall_mota:.4f}")
    logger.info(f"Mean Average Precision (mAP): {mAP:.4f}")

    if save_plots:
        # Plot per-frame metrics including mAP (mAP is overall, not per-frame)
        frames = list(range(len(per_frame_precision)))

        plt.figure(figsize=(12, 8))
        plt.plot(frames, per_frame_precision, label="Precision", marker="o")
        plt.plot(frames, per_frame_recall, label="Recall", marker="s")
        plt.plot(frames, per_frame_f1, label="F1 Score", marker="^")
        plt.plot(frames, per_frame_mota, label="MOTA", marker="d")

        plt.xlabel("Frame")
        plt.ylabel("Metric Value")
        plt.title("YOLO Evaluation Metrics Per Frame")
        plt.legend()
        plt.grid(True)

        # Annotate the plot with overall metrics
        textstr = "\n".join(
            (
                f"Overall Precision: {overall_precision:.2f}",
                f"Overall Recall: {overall_recall:.2f}",
                f"Overall F1 Score: {overall_f1:.2f}",
                f"Overall MOTA: {overall_mota:.2f}",
                f"mAP: {mAP:.2f}",
            ),
        )

        props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
        plt.text(
            0.02,
            0.98,
            textstr,
            transform=plt.gca().transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=props,
        )

        per_frame_metrics_plot_path = os.path.join(save_dir, f"{seq}_per_frame_metrics.png")
        plt.savefig(per_frame_metrics_plot_path)
        logger.info(f"Saved per-frame metrics plot to {per_frame_metrics_plot_path}")
        plt.close()

        # Optionally, save a summary bar chart of overall metrics
        metrics = {
            "Precision": overall_precision,
            "Recall": overall_recall,
            "F1 Score": overall_f1,
            "MOTA": overall_mota,
            "mAP": mAP,
        }

        plt.figure(figsize=(10, 6))
        plt.bar(
            metrics.keys(),
            metrics.values(),
            color=["blue", "orange", "green", "red", "purple"],
        )
        plt.ylim(0, 1)
        plt.ylabel("Score")
        plt.title("YOLO Overall Evaluation Metrics")
        for i, (metric, value) in enumerate(metrics.items()):
            plt.text(i, value + 0.01, f"{value:.2f}", ha="center", va="bottom")
        summary_plot_path = os.path.join(save_dir, f"{seq}_overall_metrics.png")
        plt.savefig(summary_plot_path)
        logger.info(f"Saved overall metrics bar chart to {summary_plot_path}")
        plt.close()

    logger.info("YOLO model performance evaluation completed.")

    # Return or print mAP if needed
    # print(f"Mean Average Precision (mAP): {mAP:.4f}")


if __name__ == "__main__":
    import sys

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger("Main")

    # Argument parser for command-line options
    parser = argparse.ArgumentParser(description="Evaluate YOLO Model Performance")
    parser.add_argument(
        "--save_plots",
        action="store_true",
        help="Flag to save detection visualizations and evaluation plots.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="plots/yoloval",
        help="Directory to save the plots. Required if --save_plots is set.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Flag to display detection visualizations in real-time.",
    )
    args = parser.parse_args()

    # Paths (Update these paths according to your directory structure)
    config_path = "config/config.yaml"
    calibration_file = "data/34759_final_project_rect/calib_cam_to_cam.txt"
    images_left_dir = "data/34759_final_project_rect/seq_01/image_02/data/"
    images_right_dir = "data/34759_final_project_rect/seq_01/image_03/data/"
    labels_file = "data/34759_final_project_rect/seq_01/labels.txt"

    # Load configuration
    if not Path(config_path).is_file():
        logger.error(f"Configuration file not found at {config_path}")
        raise FileNotFoundError(f"Configuration file not found at {config_path}")
    with open(config_path) as file:
        config = yaml.safe_load(file)

    # Load image paths for Sequence 1
    image_paths_left = sorted(Path(images_left_dir).glob("*.png"))
    image_paths_right = sorted(Path(images_right_dir).glob("*.png"))
    image_paths_left = [str(p) for p in image_paths_left]
    image_paths_right = [str(p) for p in image_paths_right]

    # Load labels for Sequence 1
    labels = parse_labels(labels_file)

    # Load YOLO model
    yolo_model = get_model("src/models/yolov8/best.pt")  # Update the path if necessary

    # Evaluate YOLO model performance on Sequence 1
    evaluate_yolo_performance(
        model=yolo_model,
        image_paths=image_paths_left,
        labels=labels,
        iou_threshold=0.2,
        score_threshold=0.3,
        show=args.show,  # Set to True to visualize detections
        save_plots=args.save_plots,
        save_dir=args.save_dir if args.save_plots else None,
        seq="seq_01",
    )

    # Close any OpenCV windows
    cv2.destroyAllWindows()
