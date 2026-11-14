import os
import cv2
import numpy as np
import shutil
import random

# ==========================================
# 1. Project Paths & Setup
# ==========================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

INPUT_DIR = os.path.join(PROJECT_ROOT, "reeses_dataset", "cleaned")
DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset")

# YOLOv8 expected structure
SPLITS = ["train", "val"]
for split in SPLITS:
    os.makedirs(os.path.join(DATASET_DIR, f"images/{split}"), exist_ok=True)
    os.makedirs(os.path.join(DATASET_DIR, f"labels/{split}"), exist_ok=True)

# ==========================================
# 2. Reese's Orange HSV Thresholds
# ==========================================
LOWER_ORANGE = np.array([5, 120, 100])
UPPER_ORANGE = np.array([25, 255, 255])

# ==========================================
# 3. Process & Auto-Label
# ==========================================
files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith((".jpg", ".png"))]
random.seed(42)  # Fixed seed for reproducible train/val split
random.shuffle(files)

# 80% Training, 20% Validation
split_idx = int(len(files) * 0.8)
train_files = files[:split_idx]
val_files = files[split_idx:]


def process_image(filename, split):
    filepath = os.path.join(INPUT_DIR, filename)
    image = cv2.imread(filepath)
    if image is None:
        return False

    img_h, img_w = image.shape[:2]

    # Convert to HSV and mask orange color
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOWER_ORANGE, UPPER_ORANGE)

    # Remove small noise and fill gaps
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Find the candy bar contour
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False

    largest_contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest_contour) < 500:
        return False

    # Get bounding box coordinates
    x, y, w, h = cv2.boundingRect(largest_contour)

    # YOLO Format: class x_center y_center width height (Normalized 0.0 to 1.0)
    x_center = (x + (w / 2.0)) / img_w
    y_center = (y + (h / 2.0)) / img_h
    norm_w = w / float(img_w)
    norm_h = h / float(img_h)

    # Write YOLO Label .txt file
    label_filename = os.path.splitext(filename)[0] + ".txt"
    label_path = os.path.join(DATASET_DIR, f"labels/{split}", label_filename)
    with open(label_path, "w") as f:
        f.write(f"0 {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}\n")

    # Copy image to dataset structure
    dest_img_path = os.path.join(DATASET_DIR, f"images/{split}", filename)
    shutil.copy(filepath, dest_img_path)

    # Preview Window
    display = image.copy()
    cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(display, "Reese's Auto-Labeled", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.imshow("Auto-Label Preview", display)

    return True


print(f"Auto-labeling {len(files)} images...")
success_count = 0

for filename in train_files:
    if process_image(filename, "train"):
        success_count += 1
    cv2.waitKey(20)

for filename in val_files:
    if process_image(filename, "val"):
        success_count += 1
    cv2.waitKey(20)

cv2.destroyAllWindows()

# ==========================================
# 4. Generate data.yaml with Absolute Root
# ==========================================
dataset_abs_path = os.path.abspath(DATASET_DIR).replace("\\", "/")
yaml_content = f"""path: {dataset_abs_path}
train: images/train
val: images/val

names:
  0: reeses
"""
with open(os.path.join(DATASET_DIR, "data.yaml"), "w") as f:
    f.write(yaml_content)

print(f"\nSuccessfully labeled and packaged {success_count}/{len(files)} images.")
print(f"Dataset structure and data.yaml generated at: '{DATASET_DIR}'")
