import cv2
import time
import joblib
import numpy as np
import mediapipe as mp
from pathlib import Path
from collections import deque, Counter

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "word_model.joblib"
ENCODER_PATH = BASE_DIR / "models" / "label_encoder.joblib"
TASK_MODEL_PATH = BASE_DIR / "models" / "hand_landmarker.task"

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]


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
    pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)

    wrist = pts[0].copy()
    pts = pts - wrist

    scale = np.linalg.norm(pts[9])  # wrist -> middle mcp
    if scale > 1e-6:
        pts = pts / scale

    return pts.flatten().reshape(1, -1)  # (1, 63)


def most_common(items):
    return Counter(items).most_common(1)[0][0]


def main():
    if not MODEL_PATH.exists() or not ENCODER_PATH.exists():
        raise FileNotFoundError("Model files not found. Run train_quick.py first.")

    if not TASK_MODEL_PATH.exists():
        raise FileNotFoundError(f"HandLandmarker model not found at: {TASK_MODEL_PATH}")

    model = joblib.load(MODEL_PATH)
    le = joblib.load(ENCODER_PATH)

    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(TASK_MODEL_PATH)),
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

    history = deque(maxlen=10)

    print("Live test running... Press ESC to quit.")

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

            label_text = "..."
            conf = 0.0

            if result.hand_landmarks:
                hand = result.hand_landmarks[0]
                draw_hand(frame, hand)

                X = normalize_landmarks(hand)

                proba = model.predict_proba(X)[0]
                pred_idx = int(np.argmax(proba))
                conf = float(proba[pred_idx])
                pred_label = le.inverse_transform([pred_idx])[0]

                # Confidence threshold (prevents random nonsense)
                if conf > 0.55:
                    history.append(pred_label)
                else:
                    history.append("...")

                label_text = most_common(history)

            cv2.putText(frame, f"Pred: {label_text}", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 3)
            cv2.putText(frame, f"Conf: {conf:.2f}", (20, 95),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

            cv2.imshow("Live Word Detection", frame)

            if (cv2.waitKey(1) & 0xFF) == 27:
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
