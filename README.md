# 🛸 Tello Reese's Peanut Butter Bar Tracker

Autonomous detection and visual tracking of a Reese's Peanut Butter Bar using a Ryze Tello drone equipped with a 45° downward-facing mirror and fine-tuned YOLOv8.

---

## 📁 Project Directory Structure

```text
tello_reeses_tracker/
├── calibration/                 # Lens undistortion calibration matrices
│   ├── camera_matrix.npy
│   └── dist_coeffs.npy
│
├── 1_data_collection/           # Step 1: Capture, filter, and label training data
│   ├── capture_dataset.py       # Captures frames from Tello (SPACE to save, Q to quit)
│   ├── clean_dataset.py         # Multi-patch Laplacian blur filter
│   └── auto_label.py            # HSV contour auto-labeler & train/val split generator
│
├── 2_model_training/            # Step 2: YOLOv8 model training
│   ├── train_model.py           # Trains YOLOv8n on local GPU (RTX 4060)
│   └── yolov8n.pt               # Base pre-trained model weights
│
├── 3_model_testing/             # Step 3: Test dataset capture & evaluation
│   ├── capture_test_data.py     # Captures independent test photos
│   └── run_test_data.py         # Runs test evaluation and saves predictions
│
├── 4_live_tracking/             # Step 4: Live deployment & tracking HUD
│   └── detect_live.py           # Real-time drone detection with tracking error vectors
│
├── dataset/                     # YOLOv8 formatted training dataset
│   ├── data.yaml
│   ├── images/ (train, val)
│   └── labels/ (train, val)
│
├── reeses_dataset/              # Image storage repository
│   ├── cleaned/                 # Cleaned, high-quality images
│   ├── images/                  # Staging for newly captured raw images
│   └── rejected/                # Blurry or clipped photos
│
├── test_dataset/                # Independent testing images & prediction outputs
│   ├── images/                  # Test photos captured from the drone
│   └── predictions/             # Annotated test predictions with bounding boxes
│
└── runs/                        # Training outputs, best.pt checkpoints, and metric graphs
```

---

## 🚀 How to Run the Workflow

All scripts automatically resolve project root paths, so you can run them either from the root or from inside each folder.

### Step 1: Data Collection & Preparation
```bash
# Capture raw training images from Tello
python 1_data_collection/capture_dataset.py

# Filter out motion-blurred frames
python 1_data_collection/clean_dataset.py

# Auto-label bounding boxes and split into 80% train / 20% val
python 1_data_collection/auto_label.py
```

### Step 2: Model Training
```bash
# Train YOLOv8n on GPU
python 2_model_training/train_model.py
```
*Trained weights are automatically saved to `runs/detect/reeses_detector/weights/best.pt`.*

### Step 3: Model Testing
```bash
# Capture new test images from the drone
python 3_model_testing/capture_test_data.py

# Evaluate test images and view accuracy & bounding boxes
python 3_model_testing/run_test_data.py
```

### Step 4: Live Drone Detection
```bash
# Launch live detection HUD with real-time tracking error vectors
python 4_live_tracking/detect_live.py
```
