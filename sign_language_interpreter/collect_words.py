import cv2
import time
import csv
import os
import mediapipe as mp
import numpy as np
from pathlib import Path

# ----- Your dictionary -----
LABELS = [
    "ok", "stop", "thumbs_up", "thumbs_down", "peace",
    "fist", "open_palm", "point", "call_me", "rock",
    "pinch", "love_you"
]

# Add more if you want:
# LABELS += ["three", "four", "five"]

# ----- Paths -----
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "hand_landmarker.task"
DATA_DIR = BASE_DIR / "data"
CSV_PATH = DATA_DIR / "words.csv"

# Hand skeleton connections for drawing
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]

def ensure_csv():
    DATA_DIR.mkdir(exist_ok=True)
    if not CSV_PATH.exists():
        with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            header = ["label"] + [f"f{i}" for i in range(63)]  # 21*3
            writer.writerow(header)

def draw_hand(frame, landmarks):
    h, w, _ = frame.shape
    pts = []
    for lm in landmarks:
        x = int(lm.x * w)
        y = int(lm.y * h)
        pts.append((x, y))
        cv2.circle(frame, (x, y), 4, (255, 255, 255), -1)

    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (255, 255, 255), 2)

def normalize_landmarks(landmarks):
    """
    Convert 21 landmarks to a normalized 63-length vector:
    - translation invariance: subtract wrist
    - scale invariance: divide by wrist->middle_mcp distance
    """
    pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)

    wrist = pts[0].copy()
    pts = pts - wrist

    scale = np.linalg.norm(pts[9])  # middle_mcp (rough hand size)
    if scale > 1e-6:
        pts = pts / scale

    return pts.flatten()  # shape (63,)

def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at: {MODEL_PATH}")

    ensure_csv()

    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.6,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        return

    label_idx = 0
    saved = 0

    print("Collecting data...")
    print("N=next label | B=prev label | R=record | ESC=quit")

    with HandLandmarker.create_from_options(options) as landmarker:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            timestamp_ms = int(time.time() * 1000)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            current_label = LABELS[label_idx]
            features = None
            handedness_text = ""

            if result.hand_landmarks:
                hand = result.hand_landmarks[0]
                draw_hand(frame, hand)

                features = normalize_landmarks(hand)

                if result.handedness and result.handedness[0]:
                    handedness_text = result.handedness[0][0].category_name

            # UI overlay
            cv2.putText(frame, f"Label: {current_label}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 3)

            cv2.putText(frame, f"Hand: {handedness_text}", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            cv2.putText(frame, f"Saved: {saved}", (20, 115),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            cv2.putText(frame, "R=record | N=next | B=back | ESC=quit", (20, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.imshow("Collect Words", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == 27:  # ESC
                break

            elif key in [ord("n"), ord("N")]:
                label_idx = (label_idx + 1) % len(LABELS)

            elif key in [ord("b"), ord("B")]:
                label_idx = (label_idx - 1) % len(LABELS)

            elif key in [ord("r"), ord("R")]:
                if features is not None:
                    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
                        writer = csv.writer(f)
                        writer.writerow([current_label] + features.tolist())
                    saved += 1
                    print(f"Saved sample #{saved} for '{current_label}'")
                else:
                    print("No hand detected. Nothing saved.")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
