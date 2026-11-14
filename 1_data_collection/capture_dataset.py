import os
import time
import cv2
import numpy as np
from djitellopy import Tello

# ==========================================
# 1. Project Paths & Setup
# ==========================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

SAVE_DIR = os.path.join(PROJECT_ROOT, "reeses_dataset", "images")
os.makedirs(SAVE_DIR, exist_ok=True)

existing_files = [f for f in os.listdir(SAVE_DIR) if f.lower().endswith((".jpg", ".png"))]
img_counter = len(existing_files)
print(f"Dataset directory ready: '{SAVE_DIR}' (Existing images: {img_counter})")

# ==========================================
# 2. Calibration & Undistortion Maps
# ==========================================
calib_dir = os.path.join(PROJECT_ROOT, "calibration")
cam_matrix_path = os.path.join(calib_dir, "camera_matrix.npy")
dist_coeffs_path = os.path.join(calib_dir, "dist_coeffs.npy")

camera_matrix = np.load(cam_matrix_path)
dist_coeffs = np.load(dist_coeffs_path)

w, h = 960, 720
new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
    camera_matrix, dist_coeffs, (w, h), alpha=0, newImgSize=(w, h)
)
mapx, mapy = cv2.initUndistortRectifyMap(
    camera_matrix, dist_coeffs, None, new_camera_matrix, (w, h), cv2.CV_32FC1
)

# ==========================================
# 3. Tello Connection
# ==========================================
tello = Tello()
tello.connect()
battery = tello.get_battery()
last_battery_time = time.time()
print(f"Connected to Tello. Battery: {battery}%")

tello.streamon()
print("Waiting for video stream to initialize...")
time.sleep(2)
frame_reader = tello.get_frame_read()

last_saved_time = 0
flash_banner = False

print("\n--- Controls ---")
print(" [SPACE] : Save current undistorted frame to dataset")
print(" [Q]     : Exit\n")

try:
    while True:
        raw_frame = frame_reader.frame
        if raw_frame is None:
            continue

        # 1. Convert djitellopy RGB to OpenCV BGR
        bgr_frame = cv2.cvtColor(raw_frame, cv2.COLOR_RGB2BGR)

        # 2. Undistort using precomputed maps
        undistorted = cv2.remap(bgr_frame, mapx, mapy, cv2.INTER_LINEAR)

        # 3. Flip horizontally to correct physical mirror orientation
        final_frame = cv2.flip(undistorted, 1)

        # Create display copy so HUD doesn't save into dataset images
        display_frame = final_frame.copy()

        current_time = time.time()
        if current_time - last_battery_time > 5.0:
            try:
                battery = tello.get_battery()
            except Exception:
                pass
            last_battery_time = current_time

        if flash_banner and (current_time - last_saved_time < 0.5):
            cv2.rectangle(display_frame, (0, 0), (w, 60), (0, 200, 0), -1)
            cv2.putText(
                display_frame,
                f"SAVED: reeses_{img_counter - 1:04d}.jpg",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )
        else:
            flash_banner = False
            cv2.putText(
                display_frame,
                f"Images Captured: {img_counter} | Battery: {battery}%",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2,
            )
            cv2.putText(
                display_frame,
                "Press [SPACE] to capture | [Q] to quit",
                (20, 680),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (200, 200, 200),
                1,
            )

        cv2.imshow("Tello Dataset Collector", display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 32:  # ASCII 32 = Spacebar
            filename = os.path.join(SAVE_DIR, f"reeses_{img_counter:04d}.jpg")
            cv2.imwrite(filename, final_frame)
            img_counter += 1
            flash_banner = True
            last_saved_time = time.time()
            print(f"Captured: {filename}")

        elif key == ord("q"):
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
    print(f"\nSession finished. Total images in folder: {img_counter}")
