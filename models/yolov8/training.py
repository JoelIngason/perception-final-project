from ultralytics import YOLO

# Load the YOLOv8 model with pretrained weights, and reset the classification head for custom classes
model = YOLO("yolov8n.pt")

# Adjust the model to match the custom number of classes
model.model.nc = 9  # Set this to the number of classes defined in data.yaml (9 in your case)
model.model.names = ['bus', 'camper', 'car', 'cyclist', 'motorcycle', 'person', 'tractor', 'truck', 'van']  # Custom class names

# Train the model using the custom dataset
model.train(
    data="data.yaml",    # Path to your data.yaml file
    epochs=3,           # Number of epochs (adjust as needed)
    imgsz=640,           # Image size (adjust if you need larger images)
    batch=48,            # Batch size (adjust based on available memory)
    device=0,            # Use "0" for GPU; change to "cpu" if using CPU
    single_cls=False     # Ensures the model trains on multiple classes as per your dataset
)

# Save the trained model to a file
model.save("trained_model.pt")
