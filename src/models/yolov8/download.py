# Import necessary libraries
import os
import shutil

import yaml
from roboflow import Roboflow

# =======================
# Configuration Parameters
# =======================
ROBLOFFLOW_API_KEY = "2syVdXYlMuYRZCFyFAfp"

# Define the datasets with their respective workspace names, project names, and version numbers
DATASETS = [
    {
        "workspace": "simone-bernabe",
        "name": "smart-city-pedestrian-bike-detection",
        "version": 23,
        "classes_map": {
            "person": "Pedestrian",
            "vehicles": "Car",
            "cyclist": "Cyclist",
        },
        "filter_classes": ["person", "vehicles", "cyclist"],
    },
    {
        "workspace": "cjff",
        "name": "smart-city-pedestrian-bike-detection-s4skz",
        "version": 1,
        "classes_map": {
            "person": "Pedestrian",
            "vehicles": "Car",
            "cyclist": "Cyclist",
        },
        "filter_classes": ["person", "vehicles", "cyclist"],
    },
]

# Desired output classes
DESIRED_CLASSES = ["Pedestrian", "Car", "Cyclist"]

# Directory to store the merged dataset
MERGED_DATASET_DIR = "merged_dataset"

# Path for the YOLOv8 data config YAML
YOLOV8_DATA_CONFIG = "data.yaml"

# =========================
# Initialize Roboflow Instance
# =========================

rf = Roboflow(api_key=ROBLOFFLOW_API_KEY)

# =========================
# Function to Download and Process Each Dataset
# =========================


def download_and_process_dataset(dataset_info, output_dir):
    """
    Download a dataset from Roboflow, maps class labels, and filters classes if needed.

    Args:
        dataset_info (dict): Information about the dataset.
        output_dir (str): Directory to save the processed dataset.

    """
    # Access the specified workspace and project
    project = rf.workspace(dataset_info["workspace"]).project(dataset_info["name"])

    # Download the dataset in YOLOv8 format
    version_slug = project.version(dataset_info["version"]).name.replace(" ", "-")
    dataset_loc = f"{version_slug}-{project.version(dataset_info['version']).version}"
    print(f"Downloading dataset: {dataset_loc}")
    # Check if the dataset is already downloaded
    if os.path.exists(dataset_loc):
        print(f"Dataset already downloaded: {dataset_loc}")
    else:
        dataset = project.version(dataset_info["version"]).download("yolov8")
        dataset_loc = dataset.location

    # Path to the downloaded dataset
    dataset_path = dataset_loc

    # Define paths for images and labels
    split = ["train", "valid", "test"]
    images_dirs = [os.path.join(dataset_path, s, "images") for s in split]
    labels_dirs = [os.path.join(dataset_path, s, "labels") for s in split]
    # Make sure the images and labels directories exist
    for images_dir, labels_dir in zip(images_dirs, labels_dirs, strict=False):
        if not os.path.exists(images_dir):
            print(f"Images directory not found: {images_dir}. Skipping.")
            return
        if not os.path.exists(labels_dir):
            print(f"Labels directory not found: {labels_dir}. Skipping.")
            return

    # Create output directories for each split
    for d in split:
        os.makedirs(os.path.join(output_dir, d, "images"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, d, "labels"), exist_ok=True)

    # Read class names from Roboflow's data.yaml
    with open(os.path.join(dataset_path, "data.yaml")) as f:
        data_yaml = yaml.safe_load(f)
    class_names = data_yaml["names"]

    # Create a mapping from original class indices to new class names
    class_mapping = dataset_info.get("classes_map", {})
    filter_classes = dataset_info.get("filter_classes", None)

    # Determine the classes to keep
    classes_to_keep = filter_classes if filter_classes else list(class_mapping.keys())

    # Create a new classes list based on desired mapping
    new_class_names = []
    class_id_mapping = {}
    for original_class in classes_to_keep:
        if original_class in class_mapping:
            new_class = class_mapping[original_class]
            if new_class in DESIRED_CLASSES:
                try:
                    original_class_id = class_names.index(original_class)
                except ValueError:
                    print(
                        f"Class '{original_class}' not found in classes.txt of dataset '{dataset_info['name']}'. Skipping.",
                    )
                    continue
                new_class_id = DESIRED_CLASSES.index(new_class)
                class_id_mapping[original_class_id] = new_class_id
                if new_class not in new_class_names:
                    new_class_names.append(new_class)

    # Process each label file for each split
    for images_dir, labels_dir in zip(images_dirs, labels_dirs, strict=False):
        for label_file in os.listdir(labels_dir):
            if not label_file.endswith(".txt"):
                continue

            # Read the label file
            with open(os.path.join(labels_dir, label_file)) as f:
                lines = f.readlines()

            # Filter out labels for unwanted classes
            new_lines = []
            for line in lines:
                class_id, *rest = line.split(" ")
                class_id = int(class_id)
                if class_id in class_id_mapping:
                    new_class_id = class_id_mapping[class_id]
                    new_line = f"{new_class_id} {' '.join(rest)}"
                    new_lines.append(new_line)

            # Skip the label file if no labels are left
            if not new_lines:
                continue

            out_type = (
                "train" if "train" in labels_dir else "valid" if "valid" in labels_dir else "test"
            )
            # Write the new label file
            # Create the directory if it doesn't exist
            os.makedirs(os.path.join(output_dir, out_type, "labels"), exist_ok=True)
            with open(os.path.join(output_dir, out_type, "labels", label_file), "w") as f:
                f.write("".join(new_lines))

    # Copy images to the corresponding split directories
    for images_dir in images_dirs:
        for image_file in os.listdir(images_dir):
            out_type = (
                "train" if "train" in images_dir else "valid" if "valid" in images_dir else "test"
            )
            shutil.copyfile(
                os.path.join(images_dir, image_file),
                os.path.join(output_dir, out_type, "images", image_file),
            )

    print(f"Processed dataset: {dataset_info['name']}")


# =========================
# Download and Process All Datasets
# =========================


def prepare_merged_dataset():
    """
    Downloads all datasets, processes them, and merges into a single dataset directory.
    """
    print("Preparing merged dataset...")

    # Create merged dataset directories for train, valid, test
    for split in ["train", "valid", "test"]:
        split_images_dir = os.path.join(MERGED_DATASET_DIR, split, "images")
        split_labels_dir = os.path.join(MERGED_DATASET_DIR, split, "labels")
        os.makedirs(split_images_dir, exist_ok=True)
        os.makedirs(split_labels_dir, exist_ok=True)

    for dataset_info in DATASETS:
        download_and_process_dataset(dataset_info, MERGED_DATASET_DIR)

    print("All datasets have been merged successfully.")


# =========================
# Create YOLOv8 Data Config
# =========================


def create_data_config():
    """
    Creates a YOLOv8 data configuration YAML file.
    """
    print("Creating YOLOv8 data configuration...")
    data_yaml_content = f"""
path: {os.path.abspath(MERGED_DATASET_DIR)}
train: train/images
val: valid/images
test: test/images

nc: {len(DESIRED_CLASSES)}
names: {DESIRED_CLASSES}
"""
    with open(YOLOV8_DATA_CONFIG, "w") as f:
        f.write(data_yaml_content.strip())

    print(f"YOLOv8 data config saved to {YOLOV8_DATA_CONFIG}")


# =========================
# Main Execution
# =========================

if __name__ == "__main__":
    # Step 1: Prepare the merged dataset
    prepare_merged_dataset()

    # Step 2: Create the YOLOv8 data configuration file
    create_data_config()
