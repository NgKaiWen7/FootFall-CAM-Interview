python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt

.venv/bin/python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu129
echo "Running initial human detection"
.venv/bin/python human_segmentation.py yolo11n-seg.pt sample.mp4 human_seg

echo "Finished human detection on frames without staff id"
echo "Starting data augmentation on staff id"
.venv/bin/python id_data_augmentation.py human_seg 
echo "Finished id data augmentation"
.venv/bin/python form_id_dataset.py human_seg id
echo "Finished staff id dataset preparation"
echo "Training YOLO model on augmented staff id dataset"
.venv/bin/python train_yolo_tag_model.py yolo11n.pt human_seg id_classifier

echo "Implementing the trained YOLO model to detect the staff"
echo "Further crop out the staff for model training"
.venv/bin/python yolo_detect_human_id.py yolo11n-seg.pt id_classifier sample.mp4 staff_id
.venv/bin/python form_id_dataset.py staff_id staff
.venv/bin/python train_yolo_staff_model.py staff_detection_model yolo11n.pt staff_id
echo "Finished training staff detection model"

echo "Implementing the staff detection model to crop out the staff"
.venv/bin/python yolo_detect_tagged_human.py staff_detection_model sample.mp4 output_staff.mp4
