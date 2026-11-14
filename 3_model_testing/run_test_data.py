import os
import cv2
import numpy as np
from ultralytics import YOLO

# ==========================================
# 1. Project Paths & Setup
# ==========================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

TEST_IMAGES_DIR = os.path.join(PROJECT_ROOT, "test_dataset", "images")
TEST_LABELS_DIR = os.path.join(PROJECT_ROOT, "test_dataset", "labels")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "test_dataset", "predictions")
MODEL_PATH = os.path.join(PROJECT_ROOT, "runs", "detect", "reeses_detector", "weights", "best.pt")
CONF_THRESHOLD = 0.45  # Optimal F1 threshold

os.makedirs(OUTPUT_DIR, exist_ok=True)

if not os.path.exists(MODEL_PATH):
    print(f"Error: Model weights not found at '{MODEL_PATH}'. Train the model first!")
    exit(1)

test_files = sorted([f for f in os.listdir(TEST_IMAGES_DIR) if f.lower().endswith((".jpg", ".png", ".jpeg"))])
if not test_files:
    print(f"No test images found in '{TEST_IMAGES_DIR}'. Run '3_model_testing/capture_test_data.py' first!")
    exit(0)

# ==========================================
# 2. Load Model & Evaluate Test Images
# ==========================================
print(f"Loading custom model from '{MODEL_PATH}'...")
model = YOLO(MODEL_PATH)

print(f"\nEvaluating {len(test_files)} test images at Confidence >= {CONF_THRESHOLD:.2f}...\n")

tp = 0  # True Positives: Reese's present & detected
fp = 0  # False Positives: Empty/negative frame, but Reese's detected
fn = 0  # False Negatives: Reese's present, but missed
tn = 0  # True Negatives: Empty/negative frame & correctly not detected

confidences = []
missed_images = []
false_alarm_images = []

print(f"{'Image Name':<20} | {'Expected':<10} | {'Detected':<10} | {'Confidence':<12} | {'Result':<10}")
print("-" * 72)

for filename in test_files:
    img_path = os.path.join(TEST_IMAGES_DIR, filename)
    image = cv2.imread(img_path)
    if image is None:
        continue

    # Ground truth determination:
    # Files starting with 'neg_', 'empty_', 'bg_', or 'distractor_' are treated as Negative (no candy bar).
    # All other files are treated as Positive (candy bar present).
    is_expected_positive = not any(
        filename.lower().startswith(prefix) for prefix in ["neg_", "empty_", "bg_", "distractor_"]
    )

    # If test_dataset/labels exists with a .txt file, respect that annotation
    label_file = os.path.join(TEST_LABELS_DIR, os.path.splitext(filename)[0] + ".txt")
    if os.path.exists(label_file):
        with open(label_file, "r") as f:
            lines = [line.strip() for line in f if line.strip()]
            is_expected_positive = len(lines) > 0

    # Run inference on GPU (device=0)
    results = model.predict(source=image, conf=CONF_THRESHOLD, device=0, verbose=False)[0]
    boxes = results.boxes
    has_detection = len(boxes) > 0

    # Classify outcome
    if is_expected_positive and has_detection:
        outcome = "TP (Hit)"
        tp += 1
    elif is_expected_positive and not has_detection:
        outcome = "FN (Miss)"
        fn += 1
        missed_images.append(filename)
    elif not is_expected_positive and has_detection:
        outcome = "FP (Alarm)"
        fp += 1
        false_alarm_images.append(filename)
    else:
        outcome = "TN (Clean)"
        tn += 1

    # Record confidence
    if has_detection:
        best_box = boxes[0]
        conf = float(best_box.conf[0].cpu().numpy())
        confidences.append(conf)
        conf_str = f"{conf:.2%}"
    else:
        conf_str = "---"

    exp_str = "Reese's" if is_expected_positive else "Empty"
    det_str = "Reese's" if has_detection else "None"

    print(f"{filename:<20} | {exp_str:<10} | {det_str:<10} | {conf_str:<12} | {outcome:<10}")

    # Save visual prediction
    annotated = results.plot()
    output_path = os.path.join(OUTPUT_DIR, f"pred_{filename}")
    cv2.imwrite(output_path, annotated)

# ==========================================
# 3. Compute Metrics
# ==========================================
total_samples = tp + tn + fp + fn
accuracy = ((tp + tn) / total_samples) * 100 if total_samples > 0 else 0.0
precision = (tp / (tp + fp)) * 100 if (tp + fp) > 0 else 0.0
recall = (tp / (tp + fn)) * 100 if (tp + fn) > 0 else 0.0
f1_score = (2 * (precision * recall) / (precision + recall)) if (precision + recall) > 0 else 0.0

avg_conf = (sum(confidences) / len(confidences)) * 100 if confidences else 0.0

# ==========================================
# 4. Display Formatted Report & Confusion Matrix
# ==========================================
print("\n" + "=" * 60)
print("             TEST DATASET EVALUATION REPORT             ")
print("=" * 60)
print(f"Total Test Images      : {total_samples}")
print(f"Accuracy               : {accuracy:.2f}%")
print(f"Precision              : {precision:.2f}%")
print(f"Recall                 : {recall:.2f}%")
print(f"F1 Score               : {f1_score:.2f}%")
print(f"Average Confidence     : {avg_conf:.2f}%")
if confidences:
    print(f"Confidence Range       : Min {min(confidences):.2%}  |  Max {max(confidences):.2%}")

print("\n" + "-" * 60)
print("                     CONFUSION MATRIX                    ")
print("-" * 60)
print(f"{'':<20} | {'Predicted: Reese':<18} | {'Predicted: None':<15}")
print("-" * 60)
print(f"{'Actual: Reese':<20} | {f'TP = {tp}':<18} | {f'FN = {fn}':<15}")
print(f"{'Actual: Empty/Neg':<20} | {f'FP = {fp}':<18} | {f'TN = {tn}':<15}")
print("-" * 60)

if missed_images:
    print(f"\nMissed Images (FN)     : {missed_images}")
if false_alarm_images:
    print(f"False Alarms (FP)      : {false_alarm_images}")

print(f"\nAnnotated prediction images saved to: '{OUTPUT_DIR}/'")
print("=" * 60 + "\n")
