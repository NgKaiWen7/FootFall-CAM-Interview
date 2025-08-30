from ultralytics import YOLO
import cv2
import numpy as np
import os
import shutil

model = YOLO("yolo11n-seg.pt")
video_path = "sample.mp4"
cap = cv2.VideoCapture(video_path)

if os.path.exists("human_seg"):
    shutil.rmtree("human_seg")
os.mkdir("human_seg")

if os.path.exists("human_seg_dataset"):
    shutil.rmtree("human_seg_dataset")
os.mkdir("human_seg_dataset")


cropped_humans = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, classes=0, conf=0.5, verbose=False)
    display_frame = np.zeros_like(frame)

    if results[0].masks is not None:
        masks = results[0].masks.data.cpu().numpy()
        boxes = results[0].boxes.xyxy.cpu().numpy()

        for i, (mask, box) in enumerate(zip(masks, boxes)):
            mask = (mask > 0).astype(np.uint8) * 255
            mask = cv2.resize(
                mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST
            )
            human_crop = cv2.bitwise_and(frame.copy(), frame.copy(), mask=mask)
            x1, y1, x2, y2 = map(int, box)
            human_crop = human_crop[y1:y2, x1:x2]
            cv2.imwrite(f"human_seg/{cropped_humans}.jpg", human_crop)
            np.save(f"human_seg/{cropped_humans}.npy", mask)
            cropped_humans += 1

cap.release()
cv2.destroyAllWindows()
