import cv2
import time
import mediapipe as mp
import numpy as np
from pathlib import Path

# Same connections we used
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]

MODEL_PATH = Path(__file__).resolve().parent / "models" / "hand_landmarker.task"


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


def landmarks_to_array(landmarks):
    # 21 landmarks, each has x,y,z
    return np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at: {MODEL_PATH}")

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

    print("Landmarks live... Press ESC to quit.")
    print("Tip: Look at console output for numbers.")

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

            if result.hand_landmarks:
                hand = result.hand_landmarks[0]
                draw_hand(frame, hand)

                arr = landmarks_to_array(hand)

                # Print wrist + index fingertip just as a preview
                wrist = arr[0]
                index_tip = arr[8]
                print(f"Wrist: {wrist} | Index Tip: {index_tip}")

            cv2.imshow("Landmarks Live", frame)
            if (cv2.waitKey(1) & 0xFF) == 27:
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
