from ultralytics import YOLO
import cv2
import numpy as np
import os
import shutil
import sys

if len(sys.argv) == 4:
    model = sys.argv[1]
    source_video = sys.argv[2]
    output_path = sys.argv[3]

else:
    raise ValueError("Insufficient arguments provided")


model = YOLO(model)
cap = cv2.VideoCapture(source_video)
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

if os.path.exists(output_path):
    shutil.rmtree(output_path)
os.mkdir(output_path)


def time_frame_of_intrest(t):
    time_ranges = [(0, 14), (22, 31), (49, np.inf)]
    for start, end in time_ranges:
        if start < t and t < end:
            return True
    return False


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


cropped_humans = 0
included_locations = np.empty((0, 5))
while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, classes=0, conf=0.5, verbose=False)
    display_frame = np.zeros_like(frame)
    t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000

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
            new_location = np.array([x1, y1, x2, y2, t])
            if len(included_locations) == 0:
                included_locations = np.vstack([included_locations, [new_location]])
            if time_frame_of_intrest(t) and rect_intersection_area(
                included_locations, new_location, 0.5
            ):
                included_locations = np.vstack([included_locations, [new_location]])
                cv2.imwrite(f"{output_path}/{cropped_humans}.jpg", human_crop)
                mask = mask[y1:y2, x1:x2]
                np.save(f"{output_path}/{cropped_humans}.npy", mask)
                cropped_humans += 1
                included_locations = included_locations[
                    t - included_locations[:, 4] < 1
                ]

cap.release()
cv2.destroyAllWindows()
