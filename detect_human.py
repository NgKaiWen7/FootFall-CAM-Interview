from ultralytics import YOLO
import os
import cv2
import shutil
import numpy as np
import sys


if len(sys.argv) == 4:
    yolo_model = sys.argv[1]
    source_video = sys.argv[2]
    output_video = sys.argv[3]
else:
    raise ValueError("Image source must be provided")

person_model = YOLO(yolo_model)
cap = cv2.VideoCapture(source_video)


def rect_intersection_area(boxes, ref, thresh_hold):
    boxes = np.array(boxes)
    xA = np.maximum(boxes[:, 0], ref[0])
    yA = np.maximum(boxes[:, 1], ref[1])
    xB = np.minimum(boxes[:, 2], ref[2])
    yB = np.minimum(boxes[:, 3], ref[3])
    inter_w = np.maximum(0, xB - xA)
    inter_h = np.maximum(0, yB - yA)
    area = inter_w * inter_h
    ref_area = (ref[2] - ref[0]) * (ref[3] - ref[1])
    if np.any(area / ref_area > thresh_hold):
        return False
    else:
        return True


def time_frame_of_intrest(t):
    time_ranges = [(0, 14), (22, 31), (49, np.inf)]
    for start, end in time_ranges:
        if start < t and t < end:
            return True
    return False


fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(
    output_video,
    fourcc,
    cap.get(cv2.CAP_PROP_FPS),
    (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))),
)

human_frame_idx = 0
frame_count = 0
image_crop_path = "human_crops"
if os.path.exists(image_crop_path):
    shutil.rmtree(image_crop_path)
os.makedirs(image_crop_path, exist_ok=True)

included_locations = np.empty((0, 5))
while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    frame_display = frame.copy()
    human_results = person_model(frame, classes=0, verbose=False)
    t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000

    if len(human_results[0].boxes) == 0:
        continue
    for box in human_results[0].boxes.xyxy.cpu().numpy():
        x1, y1, x2, y2 = map(int, box[:4])

        new_location = np.array([x1, y1, x2, y2, t])
        if len(included_locations) == 0:
            included_locations = np.vstack([included_locations, [new_location]])

        if time_frame_of_intrest(t) and rect_intersection_area(
            included_locations, new_location, 0.5
        ):
            included_locations = np.vstack([included_locations, [new_location]])
            img_filename = f"{image_crop_path}/frame_{human_frame_idx}.jpg"
            cv2.imwrite(img_filename, frame[y1:y2, x1:x2])
            human_frame_idx += 1
            included_locations = included_locations[t - included_locations[:, 4] < 1]

        cv2.rectangle(
            frame_display,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2,
        )

        # can be % 5 also producing good results
        if frame_count % 5 == 0 and (t):
            pass
            # human_frame_idx += 1
    frame_count += 1
    out.write(frame_display)

cap.release()
cv2.destroyAllWindows()
