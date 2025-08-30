from ultralytics import YOLO
import cv2
import sys

if len(sys.argv) == 4:
    yolo_model = sys.argv[1]
    staff_detection_model = sys.argv[1]
    target_video = sys.argv[2]
    output_video = sys.argv[3]
else:
    raise ValueError("Image source must be provided")

person_model = YOLO(f"{staff_detection_model}/train/weights/best.pt")

cap = cv2.VideoCapture(target_video)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(
    output_video,
    fourcc,
    cap.get(cv2.CAP_PROP_FPS),
    (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))),
)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    frame_display = frame.copy()

    results = person_model(frame, verbose=False)

    ## for debugging
    # if len(results[0].boxes) > 0:
    # print(f"Detected {len(results[0].boxes)} object(s) in this frame")
    annotated = results[0].plot()
    out.write(annotated)

cap.release()
cv2.destroyAllWindows()
