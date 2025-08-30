import random
import shutil
import os
import sys

if len(sys.argv) == 2:
    image_source = sys.argv[1]
else:
    raise ValueError("Image source must be provided")

output_dir = f"{image_source}_dataset_split"
image_source_dir = f"{image_source}_dataset"
train_ratio = 0.8

splits = ["train", "val"]
classes = ["id", "no_id"]

if os.path.exists(output_dir):
    shutil.rmtree(output_dir)


# make output folders
for split in splits:
    for cls in classes:
        os.makedirs(os.path.join(output_dir, split, cls), exist_ok=True)

for cls in classes:
    cls_path = os.path.join(image_source_dir, cls)
    files = os.listdir(cls_path)
    random.shuffle(files)

    split_idx = int(len(files) * train_ratio)
    train_files = files[:split_idx]
    val_files = files[split_idx:]

    for f in train_files:
        shutil.copy(
            os.path.join(cls_path, f), os.path.join(output_dir, "train", cls, f)
        )
    for f in val_files:
        shutil.copy(os.path.join(cls_path, f), os.path.join(output_dir, "val", cls, f))
