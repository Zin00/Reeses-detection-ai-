import os
from ultralytics import YOLO

# ==========================================
# 1. Project Paths & Setup
# ==========================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

DATA_YAML = os.path.join(PROJECT_ROOT, "dataset", "data.yaml")
BASE_WEIGHTS = os.path.join(SCRIPT_DIR, "yolov8n.pt")
RUNS_DIR = os.path.join(PROJECT_ROOT, "runs", "detect")


def main():
    if not os.path.exists(DATA_YAML):
        print(f"Error: Dataset config not found at '{DATA_YAML}'. Run auto_label.py first!")
        exit(1)

    # Use local yolov8n.pt if present, or download automatically
    model_source = BASE_WEIGHTS if os.path.exists(BASE_WEIGHTS) else "yolov8n.pt"
    print(f"Loading base YOLOv8 model: {model_source}")
    model = YOLO(model_source)

    print("Starting YOLOv8 training on NVIDIA RTX 4060...")

    # Train model
    results = model.train(
        data=DATA_YAML,
        epochs=50,
        imgsz=640,
        batch=16,
        device=0,  # RTX 4060 GPU
        project=RUNS_DIR,
        name="reeses_detector",
        exist_ok=True,  # Overwrite or update model in reeses_detector
        workers=2,
    )

    best_weights = os.path.join(RUNS_DIR, "reeses_detector", "weights", "best.pt")
    print("\nTraining Complete!")
    print(f"Custom trained weights saved at: {best_weights}")


if __name__ == "__main__":
    main()
