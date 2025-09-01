from ultralytics import YOLO
import cv2
import sys
import numpy as np
import os
import shutil

if len(sys.argv) == 5:
    yolo_model = sys.argv[1]
    classifier_model = sys.argv[2]
    target_video = sys.argv[3]
    staff_dataset_output = sys.argv[4]
else:
    raise ValueError("Insufficient arguments provided")

person_model = YOLO(yolo_model)
tag_model = YOLO(f"{classifier_model}/train/weights/best.pt")

video_path = "sample.mp4"
cap = cv2.VideoCapture(video_path)


def rect_intersection_area(r1, r2):
    x1_min, y1_min, x1_max, y1_max = r1
    x2_min, y2_min, x2_max, y2_max = r2
    r2_area = (x2_max - x2_min) * (y2_max - y2_min)

    x_overlap = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
    y_overlap = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))

    return x_overlap * y_overlap / r2_area


def check_previous(previous_tagged_locations, new_location):
    for frames in previous_tagged_locations[::-1]:
        if len(frames) == 0:
            continue
        for loc in frames:
            intersection = rect_intersection_area(loc, new_location)
            if intersection > 0.5:
                return True
    return False


if os.path.exists("detected_frames"):
    shutil.rmtree("detected_frames")
os.mkdir("detected_frames")

staff_dataset_output = f"{staff_dataset_output}_dataset"
if os.path.exists(staff_dataset_output):
    shutil.rmtree(staff_dataset_output)
os.mkdir(f"{staff_dataset_output}")
os.mkdir(f"{staff_dataset_output}/images")
os.mkdir(f"{staff_dataset_output}/labels")

detected_index = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape

    human_results = person_model(frame, classes=0, conf=0.5, verbose=False)
    if human_results[0].masks is not None:
        human_masks = human_results[0].masks.data.cpu().numpy()
        human_boxes = human_results[0].boxes.xyxy.cpu().numpy()
        for human_mask, human_box in zip(human_masks, human_boxes):
            human_mask = (human_mask > 0).astype(np.uint8) * 255
            human_mask = cv2.resize(
                human_mask,
                (frame.shape[1], frame.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

            human_crop = cv2.bitwise_and(frame.copy(), frame.copy(), mask=human_mask)
            human_x1, human_y1, human_x2, human_y2 = map(int, human_box[:4])
            if human_crop.size == 0:
                continue

            tag_results = tag_model(
                human_crop[human_y1:human_y2, human_x1:human_x2], verbose=False
            )
            tag_results = tag_results[0]
            if tag_results.boxes is None and len(tag_results.boxes) == 0:
                continue
            for box in tag_results.boxes:
                model_view = human_crop[human_y1:human_y2, human_x1:human_x2]
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = round(float(box.conf[0].data), 2)
                cls = int(box.cls[0])
                label = tag_results.names[cls]
                color = (0, 255, 0)
                cv2.rectangle(model_view, (x1, y1), (x2, y2), color, 2)
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                cv2.putText(
                    model_view,
                    f"{label}",
                    (center_x, center_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.3,
                    color,
                    2,
                )
                # for visualisation on where the tag is located
                cv2.imwrite(f"detected_frames/{detected_index} {conf}.jpg", model_view)

                # for another model to detect staff based on the image
                cv2.imwrite(
                    f"{staff_dataset_output}/images/{detected_index}.jpg",
                    frame.copy(),
                )
                with open(
                    f"{staff_dataset_output}/labels/{detected_index}.txt", "w"
                ) as f:
                    x_center = (human_x1 + human_x2) / 2 / w
                    y_center = (human_y1 + human_y2) / 2 / h
                    width = (human_x2 - human_x1) / w
                    height = (human_y2 - human_y1) / h
                    f.write(f"0 {x_center} {y_center} {width} {height}")
                detected_index += 1

cap.release()
cv2.destroyAllWindows()
