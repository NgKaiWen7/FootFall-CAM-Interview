from ultralytics import YOLO
import os
import cv2
import shutil
import sys


if len(sys.argv) == 5:
    human_model_path = sys.argv[1]
    id_classifier = sys.argv[2]
    source_video = sys.argv[3]
    dataset_folder = sys.argv[4]
else:
    raise ValueError("Image source must be provided")

# load models
person_model = YOLO(human_model_path)
tag_model = YOLO(f"{id_classifier}/train/weights/best.pt")

cap = cv2.VideoCapture(source_video)


def classify_using_efficientnet(tag_model, image):
    results = tag_model(image, verbose=False)
    probs = results[0].probs.data
    idx = probs.argmax()
    prob = probs[idx]
    label = "id" if idx == 0 else "no id"
    return prob, label


# just for visualization during development process
if os.path.exists("detected_frames"):
    shutil.rmtree("detected_frames")
os.makedirs("detected_frames")


if os.path.exists(dataset_folder):
    shutil.rmtree(dataset_folder)
os.makedirs(f"{dataset_folder}/labels")
os.makedirs(f"{dataset_folder}/images")


detected_frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    display_frame = frame.copy()

    # Run detection on tile
    human_results = person_model(frame, classes=0, verbose=False)
    for box in human_results[0].boxes.xyxy.cpu().numpy():
        x1, y1, x2, y2 = map(int, box[:4])

        # Crop the human region
        human_crop = frame[y1:y2, x1:x2]
        if human_crop.size == 0:
            continue

        probs, item = classify_using_efficientnet(tag_model, human_crop)

        if item.lower() == "id" and probs > 0.99:
            color = (0, 0, 255)  # red
            cv2.imwrite(
                f"{dataset_folder}/images/frame_{detected_frame_idx}.jpg", frame
            )
            x_center = (x1 + x2) / 2 / w
            y_center = (y1 + y2) / 2 / h
            bbox_w = (x2 - x1) / w
            bbox_h = (y2 - y1) / h
            label_path = f"{dataset_folder}/labels/frame_{detected_frame_idx}.txt"
            with open(label_path, "w") as f:
                f.write(f"0 {x_center} {y_center} {bbox_w} {bbox_h}\n")
            detected_frame_idx += 1
        else:
            color = (0, 255, 0)  # green

        # draw detected humans and humans with ID in red
        label = f"{item} {probs:.2f}"
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        cv2.putText(
            display_frame,
            label,
            (center_x, center_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )

        # just for visualization, can be removed
        if item.lower() == "id" and probs > 0.99:
            cv2.imwrite(
                f"detected_frames/frame_{detected_frame_idx}.jpg", display_frame
            )

cap.release()
cv2.destroyAllWindows()
