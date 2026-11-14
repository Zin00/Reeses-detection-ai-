import os
import time
import cv2
import numpy as np
import av
from djitellopy import Tello
from ultralytics import YOLO

# Suppress H.264 macroblock decoding warnings from Wi-Fi packet drops
av.logging.set_level(av.logging.PANIC)

# ==========================================
# 1. Project Paths & Setup
# ==========================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

MODEL_PATH = os.path.join(PROJECT_ROOT, "runs", "detect", "reeses_detector", "weights", "best.pt")
CONF_THRESHOLD = 0.45  # Optimal F1 threshold

if not os.path.exists(MODEL_PATH):
    print(f"Error: Model weights not found at '{MODEL_PATH}'! Train the model first.")
    exit(1)

# Load lens calibration matrices
calib_dir = os.path.join(PROJECT_ROOT, "calibration")
camera_matrix = np.load(os.path.join(calib_dir, "camera_matrix.npy"))
dist_coeffs = np.load(os.path.join(calib_dir, "dist_coeffs.npy"))

w, h = 960, 720
center_x, center_y = w // 2, h // 2

new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(
    camera_matrix, dist_coeffs, (w, h), alpha=0, newImgSize=(w, h)
)
mapx, mapy = cv2.initUndistortRectifyMap(
    camera_matrix, dist_coeffs, None, new_camera_matrix, (w, h), cv2.CV_32FC1
)

# Load YOLO model
print(f"Loading YOLO model from '{MODEL_PATH}'...")
model = YOLO(MODEL_PATH)

# ==========================================
# 2. Connect to Tello Drone
# ==========================================
tello = Tello()
tello.connect()
# Cache battery to prevent flooding the Tello UDP socket every frame
battery = tello.get_battery()
last_battery_time = time.time()
print(f"Connected to Tello! Battery: {battery}%")

tello.streamon()
print("Starting video stream...")
time.sleep(2)
frame_reader = tello.get_frame_read()

prev_time = time.time()
print("\n--- Live Detection Running ---")
print(" Press [Q] in the video window to quit.\n")

try:
    while True:
        raw_frame = frame_reader.frame
        if raw_frame is None:
            continue

        # 1. Convert RGB to BGR
        bgr = cv2.cvtColor(raw_frame, cv2.COLOR_RGB2BGR)

        # 2. Undistort camera lens distortion
        undistorted = cv2.remap(bgr, mapx, mapy, cv2.INTER_LINEAR)

        # 3. Horizontal flip to correct physical mirror reflection
        frame = cv2.flip(undistorted, 1)

        # 4. Run real-time YOLO inference on GPU
        t_start = time.time()
        results = model.predict(source=frame, conf=CONF_THRESHOLD, device=0, verbose=False)[0]
        infer_ms = (time.time() - t_start) * 1000

        # Draw frame center crosshair (Target alignment goal)
        cv2.drawMarker(frame, (center_x, center_y), (255, 255, 255), cv2.MARKER_CROSS, 20, 1)

        boxes = results.boxes
        target_found = len(boxes) > 0

        if target_found:
            # Pick highest confidence detection
            box = boxes[0]
            conf = float(box.conf[0].cpu().numpy())
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].cpu().numpy()]
            obj_cx = (x1 + x2) // 2
            obj_cy = (y1 + y2) // 2

            # Bounding box & target center
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, (obj_cx, obj_cy), 5, (0, 0, 255), -1)

            # Line from frame center to target center (Tracking vector)
            cv2.line(frame, (center_x, center_y), (obj_cx, obj_cy), (0, 255, 255), 2)

            # Tracking errors (in pixels)
            dx = obj_cx - center_x
            dy = obj_cy - center_y

            label = f"Reese's {conf:.2%} | dx:{dx:+d} dy:{dy:+d}"
            cv2.putText(frame, label, (x1, max(25, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            status_text = "STATUS: TARGET LOCKED"
            status_color = (0, 255, 0)
        else:
            status_text = "STATUS: SEARCHING..."
            status_color = (0, 165, 255)

        # Compute FPS
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
        prev_time = curr_time

        # Update battery once every 5 seconds (not every frame!)
        if curr_time - last_battery_time > 5.0:
            try:
                battery = tello.get_battery()
            except Exception:
                pass
            last_battery_time = curr_time

        # Telemetry HUD
        cv2.rectangle(frame, (0, 0), (w, 50), (30, 30, 30), -1)
        cv2.putText(frame, f"FPS: {fps:.1f} ({infer_ms:.1f}ms)", (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(frame, f"Battery: {battery}%", (260, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
        cv2.putText(frame, status_text, (500, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65, status_color, 2)

        cv2.imshow("Tello Reese's Real-Time Tracker", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cv2.destroyAllWindows()
    try:
        tello.streamoff()
    except Exception:
        pass
    try:
        tello.end()
    except Exception:
        pass
    print("\nLive detection session ended cleanly.")
