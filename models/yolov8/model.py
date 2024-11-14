from pathlib import Path

import cv2
import matplotlib.pyplot as plt
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLO

# Constants
SUPPORTED_LABELS = {"person", "car", "bicycle"}
LABEL_BACKGROUND_COLOR = (0, 0, 0, 0.5)  # Black with 50% opacity
BBOX_COLOR = "red"
LABEL_TEXT_COLOR = "white"
VIDEO_CODEC = 'avc1'
OUTPUT_VIDEO_EXTENSION = ".mp4"
FPS = 30


def plot_detections(img: cv2.Mat, tracked_objects: list, model: YOLO) -> None:
    """
    Visualize detections with semi-transparent labels using Matplotlib.

    Args:
        img (cv2.Mat): The image frame.
        tracked_objects (List): List of tracked objects from DeepSort.
        model (YOLO): The YOLO model instance.

    """
    plt.figure(figsize=(10, 6))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax = plt.gca()

    for obj in tracked_objects:
        if not obj.is_confirmed():
            continue

        track_id = obj.track_id
        bbox = obj.to_ltwh()
        label_id = obj.get_det_class()
        label_name = model.names[label_id] if label_id is not None and label_id < len(model.names) else "unknown"

        # Draw bounding box
        rect = plt.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3],
                             fill=False, edgecolor=BBOX_COLOR, linewidth=2)
        ax.add_patch(rect)

        # Prepare label text
        text = f"ID {track_id} - {label_name}"
        text_x, text_y = bbox[0], bbox[1]

        # Draw label background
        bg_rect = plt.Rectangle((text_x, text_y - 12), 150, 16,
                                color='black', alpha=0.5)
        ax.add_patch(bg_rect)

        # Draw label text
        plt.text(text_x, text_y, text, color=LABEL_TEXT_COLOR,
                 fontsize=10, verticalalignment="bottom")

    plt.axis("off")
    plt.tight_layout()
    plt.show()


def prepare_detections(results, model: YOLO) -> list[tuple[list[int], float, int]]:
    """
    Process YOLO detections and prepare them for DeepSort.

    Args:
        results: YOLO detection results.
        model (YOLO): The YOLO model instance.

    Returns:
        List of tuples containing bounding box, confidence, and class ID.

    """
    detections = []
    for box in results[0].boxes:
        cls = int(box.cls[0])
        label = model.names[cls]
        if label in SUPPORTED_LABELS:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            confidence = box.conf[0].item()
            bbox = [x1, y1, x2 - x1, y2 - y1]
            detections.append((bbox, confidence, cls))
    return detections


def detect_and_track(model: YOLO, image_path: Path, tracker: DeepSort) -> None:
    """
    Detect and track objects in a single image, then visualize the results.

    Args:
        model (YOLO): The YOLO model instance.
        image_path (Path): Path to the input image.
        tracker (DeepSort): The DeepSort tracker instance.

    """
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Warning: Unable to read image at {image_path}")
        return

    results = model(img)  # YOLO inference
    detections = prepare_detections(results, model)

    # Update DeepSort tracker
    tracked_objects = tracker.update_tracks(detections, frame=img)

    # Visualize results
    plot_detections(img, tracked_objects, model)


def process_video(image_dir: Path, output_path: Path, model: YOLO, tracker: DeepSort, fps: int = FPS) -> None:
    """
    Process a directory of images to perform object detection and tracking,
    then compile the results into a video.

    Args:
        image_dir (Path): Directory containing input images.
        output_path (Path): Path to save the output video.
        model (YOLO): The YOLO model instance.
        tracker (DeepSort): The DeepSort tracker instance.
        fps (int, optional): Frames per second for the output video. Defaults to FPS.

    """
    images = sorted(image_dir.glob("*.png"))
    if not images:
        print(f"No PNG images found in directory {image_dir}")
        return

    first_frame = cv2.imread(str(images[0]))
    if first_frame is None:
        print(f"Error: Unable to read the first image in {image_dir}")
        return

    height, width, _ = first_frame.shape
    video_writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*VIDEO_CODEC),
        fps,
        (width, height),
    )

    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Warning: Skipping unreadable image {img_path}")
            continue

        results = model(img, verbose=False)  # YOLO inference
        detections = prepare_detections(results, model)

        # Update tracker
        tracked_objects = tracker.update_tracks(detections, frame=img)

        # Annotate frame
        for obj in tracked_objects:
            if not obj.is_confirmed():
                continue

            track_id = obj.track_id
            bbox = obj.to_ltwh()
            class_id = obj.get_det_class()
            label = model.names[class_id] if class_id is not None and class_id < len(model.names) else "unknown"

            # Draw bounding box
            cv2.rectangle(img,
                          (int(bbox[0]), int(bbox[1])),
                          (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])),
                          (0, 255, 0), 2)

            # Draw label
            text = f"ID {track_id} - {label}"
            cv2.putText(img, text, (int(bbox[0]), int(bbox[1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        video_writer.write(img)

    video_writer.release()
    print(f"Video saved to {output_path}")


def initialize_tracker() -> DeepSort:
    """
    Initialize the DeepSort tracker with predefined parameters.

    Returns:
        DeepSort: An instance of the DeepSort tracker.

    """
    return DeepSort(max_age=30, n_init=3, nn_budget=100)


def load_model(model_path: str = "yolov8n.pt") -> YOLO:
    """
    Load a pretrained YOLO model.

    Args:
        model_path (str, optional): Path to the YOLO weights file. Defaults to "yolov8n.pt".

    Returns:
        YOLO: The loaded YOLO model instance.

    """
    if not Path(model_path).exists():
        raise FileNotFoundError(f"YOLO model file not found at {model_path}")
    return YOLO(model_path, verbose=False)


def main():
    """
    Main function to execute object detection and tracking on video frames.
    """
    try:
        # Load YOLO model
        model = load_model("yolov8n.pt")

        # Define directories
        current_dir = Path(__file__).parent.parent.parent
        image_directory = current_dir / "data" / "34759_final_project_raw" / "seq_01" / "image_02" / "data"
        output_video_path = current_dir / "models" / "yolov8" / "output" / "tracked_output.mp4"

        # Initialize DeepSort tracker
        tracker = initialize_tracker()

        # Ensure output directory exists
        output_video_path.parent.mkdir(parents=True, exist_ok=True)

        # Process video
        process_video(image_directory, output_video_path, model, tracker, fps=FPS)

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
