import cv2
import numpy as np
import glob
import random
import os
import shutil
import sys


def rotate(fg, angle):
    (h_fg, w_fg) = fg.shape[:2]
    center = (w_fg // 2, h_fg // 2)
    M = cv2.getRotationMatrix2D((w_fg // 2, h_fg // 2), angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    new_w = int((h_fg * sin) + (w_fg * cos))
    new_h = int((h_fg * cos) + (w_fg * sin))

    # Adjust rotation matrix (so the rotated image stays centered)
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]
    rotated = cv2.warpAffine(
        fg,
        M,
        (new_h, new_w),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )
    return rotated


# load PNG tag with alpha channel
tag = cv2.imread("edited_staff_id.png", cv2.IMREAD_UNCHANGED)  # shape: (h, w, 4)

## Commented out, this is for cropping out transparent edges
# alpha = tag[:, :, 3]
# ys, xs = np.where(alpha > 0)
# x_min, x_max = xs.min(), xs.max()
# y_min, y_max = ys.min(), ys.max()
# tag = tag[y_min : y_max + 1, x_min : x_max + 1, :]

if len(sys.argv) == 2:
    image_source = sys.argv[1]
else:
    raise ValueError("Image source must be provided")

frames = glob.glob(f"{image_source}/*.jpg")

image_output_dir = f"{image_source}_dataset"
if os.path.exists(image_output_dir):
    shutil.rmtree(image_output_dir)

for folder in ["id", "no_id"]:
    os.makedirs(os.path.join(image_output_dir, folder), exist_ok=True)


for i in range(1000):
    bg_path = random.choice(frames)
    # bg_path = frames[i]
    bg = cv2.imread(bg_path)

    # Random resize on bg
    x_scale = random.uniform(0.9, 1.1)
    y_scale = random.uniform(0.9, 1.1)
    bg = cv2.resize(bg, (0, 0), fx=x_scale, fy=y_scale)

    # for dataset without tag
    cv2.imwrite(f"{image_output_dir}/no_id/img_{i}.jpg", bg)
    h_bg, w_bg, _ = bg.shape

    # Random resize
    x_scale = random.uniform(0.1, 0.12)
    y_scale = random.uniform(0.1, 0.12)
    tag_edited = cv2.resize(tag, (0, 0), fx=x_scale, fy=y_scale)
    h_tag, w_tag, _ = tag_edited.shape

    # Random shear
    shear_M = np.array([[1, random.uniform(0, 0.3), 0], [random.uniform(0, 0.3), 1, 0]])
    tag_edited = cv2.warpAffine(tag_edited, shear_M, (w_tag, h_tag))

    # Random rotation
    angle = random.uniform(-180, 180)
    tag_edited = rotate(tag_edited, angle)

    # Random Blur (not implemented due to small id size)
    # ksize = random.choice([1, 3])
    # tag_edited = cv2.GaussianBlur(tag_edited, (ksize, ksize), 0)

    # get new tag shape to place on bg
    h_tag, w_tag, _ = tag_edited.shape

    # Random position
    x_offset = random.randint(0, w_bg - w_tag)
    y_offset = random.randint(0, h_bg - h_tag)

    # put into frame
    roi = bg[y_offset : y_offset + h_tag, x_offset : x_offset + w_tag]
    alpha = tag_edited[:, :, 3] / 255.0 * random.uniform(0.3, 0.4)
    for c in range(3):
        roi[:, :, c] = alpha * tag_edited[:, :, c] + (1 - alpha) * roi[:, :, c]
    bg[y_offset : y_offset + h_tag, x_offset : x_offset + w_tag] = roi

    cv2.imwrite(f"{image_output_dir}/id/img_{i}.jpg", bg)
