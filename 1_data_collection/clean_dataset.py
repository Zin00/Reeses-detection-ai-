import os
import shutil
import cv2

# ==========================================
# 1. Project Paths & Setup
# ==========================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

RAW_DIR = os.path.join(PROJECT_ROOT, "reeses_dataset", "images")
CLEAN_DIR = os.path.join(PROJECT_ROOT, "reeses_dataset", "cleaned")
TRASH_DIR = os.path.join(PROJECT_ROOT, "reeses_dataset", "rejected")

os.makedirs(CLEAN_DIR, exist_ok=True)
os.makedirs(TRASH_DIR, exist_ok=True)

# Minimum sharpness threshold for Tello 720p sensor
BLUR_THRESHOLD = 15.0

files = sorted([f for f in os.listdir(RAW_DIR) if f.lower().endswith((".jpg", ".png"))])

if not files:
    print(f"No new images found in '{RAW_DIR}' to clean.")
    exit()


def get_max_patch_sharpness(gray, grid=4):
    """
    Divides the frame into a grid of patches and returns the highest sharpness.
    This prevents smooth floors from suppressing off-center targets.
    """
    h, w = gray.shape
    gh, gw = h // grid, w // grid
    max_sharpness = 0.0
    for i in range(grid):
        for j in range(grid):
            patch = gray[i * gh:(i + 1) * gh, j * gw:(j + 1) * gw]
            score = cv2.Laplacian(patch, cv2.CV_64F).var()
            if score > max_sharpness:
                max_sharpness = score
    return max_sharpness


print(f"Automatically cleaning {len(files)} images using Multi-Patch Grid Sharpness...")
kept_count = 0
rejected_count = 0

for filename in files:
    filepath = os.path.join(RAW_DIR, filename)
    image = cv2.imread(filepath)
    if image is None:
        continue

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sharpness = get_max_patch_sharpness(gray, grid=4)

    if sharpness >= BLUR_THRESHOLD:
        shutil.move(filepath, os.path.join(CLEAN_DIR, filename))
        kept_count += 1
    else:
        shutil.move(filepath, os.path.join(TRASH_DIR, filename))
        rejected_count += 1

print("\n--- Cleaning Complete ---")
print(f"Good photos kept (moved to '{CLEAN_DIR}'): {kept_count}")
print(f"Blurry photos rejected (moved to '{TRASH_DIR}'): {rejected_count}")
