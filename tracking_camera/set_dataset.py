import os
import random
import csv

# Parameters
dataset_dir = "C:/Users/lucas_6hii5cu/Documents/datasets/tracking_camera/data"  # root dataset folder
images_dir = os.path.join(dataset_dir, "images")
labels_dir = os.path.join(dataset_dir, "labels")
output_csv = "C:/Users/lucas_6hii5cu/Documents/datasets/tracking_camera/data/train.csv"
n = 1500  # number of random samples

# Collect all image files
image_exts = [".jpg", ".jpeg", ".png"]
images = [f for f in os.listdir(images_dir) if os.path.splitext(f)[1].lower() in image_exts]

if n > len(images):
    raise ValueError(f"Requested {n} samples, but dataset only has {len(images)} images.")

# Randomly select n images
sampled_images = random.sample(images, n)

# Prepare CSV rows (just filenames)
rows = []
for img in sampled_images:
    label_file = os.path.splitext(img)[0] + ".txt"
    
    if not os.path.exists(os.path.join(labels_dir, label_file)):
        label_file = ""  # leave empty if no label exists
    
    rows.append([img, label_file])

# Write CSV
with open(output_csv, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["image", "label"])  # header
    writer.writerows(rows)

print(f"CSV file created: {output_csv} with {len(rows)} pairs (filenames only).")
