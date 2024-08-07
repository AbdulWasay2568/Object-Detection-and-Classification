from ultralytics import YOLO
import os

# Specify the directory where files should be saved (using escaped backslashes)
save_dir = r"YOUR_DIRECTORY"

# Load a model
model = YOLO("yolov8n.pt")  # Load a pretrained YOLOv5 model

# Train the model
results = model.train(data='LVIS.yaml', epochs=100, imgsz=640)# Train the model using "coco8.yaml


# Evaluate model performance on the validation set
metrics = model.val()

# Export the model to ONNX format
export_path = os.path.join(save_dir, "yolov8n.onnx")
path = model.export(format="onnx", export_path=export_path)  # Export the model to ONNX format

print("Model exported to:", path)  # Print the path where the model was exported