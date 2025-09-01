import random
import shutil
import os
import sys

if len(sys.argv) == 3:
    image_source = sys.argv[1]
    class_name = sys.argv[2]
else:
    raise ValueError("Image source must be provided")

output_dir = f"{image_source}_dataset_split"
image_source_dir = f"{image_source}_dataset"
train_ratio = 0.8

if os.path.exists(output_dir):
    shutil.rmtree(output_dir)


images = [f for f in os.listdir(f"{image_source_dir}/images") if f.endswith(".jpg")]
random.shuffle(images)

n = len(images)
n_train = int(0.8 * n)
n_val = int(0.2 * n)

split_files = {
    "train": images[:n_train],
    "val": images[n_train : n_train + n_val],
}

image_dir = f"{image_source_dir}/images"
label_dir = f"{image_source_dir}/labels"
for split in ["train", "val"]:
    os.makedirs(f"{output_dir}/images/{split}", exist_ok=True)
    os.makedirs(f"{output_dir}/labels/{split}", exist_ok=True)

    for img_file in split_files[split]:
        base = os.path.splitext(img_file)[0]
        label_file = f"{base}.txt"

        shutil.copy(
            f"{image_dir}/{img_file}", f"{output_dir}/images/{split}/{img_file}"
        )
        shutil.copy(
            f"{label_dir}/{label_file}", f"{output_dir}/labels/{split}/{label_file}"
        )
with open(f"{image_source}.yaml", "w") as f:
    f.write(f"train: {output_dir}/images/train\n")
    f.write(f"val: {output_dir}/images/val\n")
    f.write(f'nc: 1\nnames: ["{class_name}"]\n')

exit()
