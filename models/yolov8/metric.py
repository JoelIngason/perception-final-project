import csv
import cv2
import matplotlib.pyplot as plt

# Helper function to calculate Intersection over Union (IoU)
def calculate_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1_p, y1_p, x2_p, y2_p = box2
    
    inter_x1 = max(x1, x1_p)
    inter_y1 = max(y1, y1_p)
    inter_x2 = min(x2, x2_p)
    inter_y2 = min(y2, y2_p)
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2_p - x1_p) * (y2_p - y1_p)
    union_area = box1_area + box2_area - inter_area
    
    iou = inter_area / union_area if union_area != 0 else 0
    return iou

# Load tracking results with only pedestrians
def load_tracking_results(file_path):
    detections = {}
    with open(file_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame = int(row["frame"])
            if row["type"] == "person":  # Only count pedestrians
                if frame not in detections:
                    detections[frame] = []
                detections[frame].append({
                    "type": row["type"],
                    "bbox": [float(row["left"]), float(row["top"]), float(row["right"]), float(row["bottom"])],
                    "score": float(row["score"])
                })
    return detections

# Load ground truth with only pedestrians
def load_ground_truth(file_path):
    ground_truths = {}
    with open(file_path, 'r') as f:
        reader = csv.reader(f, delimiter=' ')
        for row in reader:
            frame = int(row[0])
            if row[2] == "Pedestrian":  # Only count pedestrians
                if frame not in ground_truths:
                    ground_truths[frame] = []
                ground_truths[frame].append({
                    "type": row[2],
                    "bbox": [float(row[6]), float(row[7]), float(row[8]), float(row[9])]
                })
    return ground_truths

# Visualization function for pedestrians
def visualize_detections(frame, frame_detections, frame_ground_truths, image_path):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.imshow(img_rgb)

    # Draw ground truth pedestrian boxes in green
    for gt in frame_ground_truths:
        gt_bbox = gt["bbox"]
        rect = plt.Rectangle((gt_bbox[0], gt_bbox[1]), gt_bbox[2] - gt_bbox[0], gt_bbox[3] - gt_bbox[1],
                             linewidth=2, edgecolor="green", facecolor="none")
        ax.add_patch(rect)
        ax.text(gt_bbox[0], gt_bbox[1] - 10, "GT: Pedestrian", color="green", fontsize=12, fontweight="bold")

    # Draw detected pedestrian boxes in blue or red based on classification correctness
    for det in frame_detections:
        det_bbox = det["bbox"]
        best_iou = 0.0
        
        # Calculate IoU with each ground truth pedestrian box
        for gt in frame_ground_truths:
            gt_bbox = gt["bbox"]
            iou = calculate_iou(gt_bbox, det_bbox)
            if iou > best_iou:
                best_iou = iou

        color = "blue" if best_iou > 0.5 else "red"  # Blue if IoU > 0.5, otherwise red
        rect = plt.Rectangle((det_bbox[0], det_bbox[1]), det_bbox[2] - det_bbox[0], det_bbox[3] - det_bbox[1],
                             linewidth=2, edgecolor=color, facecolor="none")
        ax.add_patch(rect)
        ax.text(det_bbox[0], det_bbox[1] - 10, f"IoU: {best_iou:.2f}", color=color, fontsize=12, fontweight="bold")
    
    plt.axis("off")
    plt.show()

# Load data and specify frames
tracking_results_file = "tracking_results.csv"
ground_truth_file = "/home/teitur/perception/Project/34759_final_project_raw/seq_01/labels.txt"
image_directory = "/home/teitur/perception/Project/34759_final_project_raw/seq_01/image_02/data"  # Image folder path

detections = load_tracking_results(tracking_results_file)
ground_truths = load_ground_truth(ground_truth_file)

# Specify frames to evaluate and visualize
frames_to_evaluate = [2, 50, 100]
for frame in frames_to_evaluate:
    frame_detections = detections.get(frame, [])
    frame_ground_truths = ground_truths.get(frame, [])
    image_path = f"{image_directory}/{str(frame).zfill(10)}.png"
    
    visualize_detections(frame, frame_detections, frame_ground_truths, image_path)
