from ultralytics import YOLO
import os
import shutil
import sys


if len(sys.argv) == 4:
    model = sys.argv[1]
    dataset = sys.argv[2]
    model_output_path = sys.argv[3]
else:
    raise ValueError("Insufficient arguments provided")

if os.path.exists(model_output_path):
    shutil.rmtree(model_output_path)

model = YOLO(model)

# Train
model.train(
    data=f"{dataset}.yaml",
    epochs=200,
    imgsz=512,
    batch=32,
    project=model_output_path,
)
