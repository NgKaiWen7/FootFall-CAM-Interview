from ultralytics import YOLO
import cv2
import sys


if len(sys.argv) == 5:
    yolo_model = sys.argv[1]
    classifier_model = sys.argv[2]
    target_video = sys.argv[3]
    output_video = sys.argv[4]
else:
    raise ValueError("Image source must be provided")

person_model = YOLO(yolo_model)
tag_model = YOLO(f"{classifier_model}/train/weights/best.pt")

video_path = "sample.mp4"
cap = cv2.VideoCapture(video_path)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(
    output_video,
    fourcc,
    cap.get(cv2.CAP_PROP_FPS),
    (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))),
)


def classify_using_efficientnet(tag_model, image):
    results = tag_model(image, verbose=False)
    probs = results[0].probs.data
    idx = probs.argmax()
    prob = probs[idx]
    label = "id" if idx == 0 else "no id"
    return prob, label


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


frame_i = 0
tagged_locations = []
while True:
    tagged_locations.append([])
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    frame_display = frame.copy()

    human_results = person_model(frame, classes=0, verbose=False)
    for box in human_results[0].boxes.xyxy.cpu().numpy():
        x1, y1, x2, y2 = map(int, box[:4])

        human_crop = frame[y1:y2, x1:x2]
        if human_crop.size == 0:
            continue

        probs, item = classify_using_efficientnet(tag_model, human_crop)

        if item.lower() == "tag" and probs > 0.95:
            color = (0, 0, 255)  # red
            tagged_locations[-1].append([x1, y1, x2, y2])
        elif check_previous(tagged_locations, (x1, y1, x2, y2)):
            color = (0, 0, 255)
            item = "tag"
        else:
            color = (0, 255, 0)  # green

        label = f"{item} {probs:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        cv2.putText(
            frame, label, (center_x, center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
        )
    tagged_locations = tagged_locations[-25:]
    out.write(frame)
    frame_i += 1

cap.release()
cv2.destroyAllWindows()
