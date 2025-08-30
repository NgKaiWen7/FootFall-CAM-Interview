from ultralytics import YOLO
import os
import shutil
import sys


if len(sys.argv) == 4:
    image_source = sys.argv[1]
    classifier_model = sys.argv[2]
    model_output_path = sys.argv[3]
else:
    raise ValueError("Image source must be provided")

if os.path.exists(model_output_path):
    shutil.rmtree(model_output_path)

model = YOLO(classifier_model)

# Train
model.train(
    data=f"{image_source}_dataset_split",
    epochs=200,
    imgsz=512,
    batch=32,
    project=model_output_path,
)
