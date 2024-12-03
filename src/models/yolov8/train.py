# Import necessary libraries
import os
import shutil

import yaml
from roboflow import Roboflow
from ultralytics import YOLO

# Path for the YOLOv8 data config YAML
YOLOV8_DATA_CONFIG = "data.yaml"

# =========================
# Train YOLOv8 Model
# =========================


def train_yolov8():
    """
    Trains the YOLOv8 model using the prepared dataset and configuration.
    """
    print("Initializing YOLOv8 model...")
    # You can choose other pretrained weights like 'yolov8s.pt', 'yolov8m.pt', etc.
    model = YOLO("yolov8m.pt")

    print("Starting training...")
    model.train(
        data=YOLOV8_DATA_CONFIG,
        epochs=50,  # Adjust the number of epochs as needed
        batch=12,  # Batch size (adjust based on your GPU memory)
        name="yolov8_pedestrian_car_cyclist",  # Name of the training run
    )
    print("Training completed.")


# =========================
# Main Execution
# =========================

if __name__ == "__main__":
    # Step 3: Train the YOLOv8 model
    train_yolov8()
