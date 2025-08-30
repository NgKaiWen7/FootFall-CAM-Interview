from ultralytics import YOLO
import os, shutil
import sys

if len(sys.argv) == 3:
    model_output_path = sys.argv[1]
    yolo_classifier = sys.argv[2]
else:
    raise ValueError("Image source must be provided")

if os.path.exists(model_output_path):
    shutil.rmtree(model_output_path)

# Load a pre-trained YOLO classifier
model = YOLO(yolo_classifier)

# Train
model.train(
    data="tagged_staff_dataset.yaml",
    epochs=200,
    imgsz=512,
    batch=32,
    project=model_output_path,
)
