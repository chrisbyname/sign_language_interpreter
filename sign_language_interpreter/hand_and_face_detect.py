import time
from pathlib import Path
import cv2
import mediapipe as mp


# ----------------------------
# Paths (relative to this file)
# ----------------------------
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

HAND_MODEL_PATH = MODELS_DIR / "hand_landmarker.task"
FACE_MODEL_PATH = MODELS_DIR / "blaze_face_short_range.tflite"


# ----------------------------
# Drawing helpers
# ----------------------------
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # Index
    (5, 9), (9, 10), (10, 11), (11, 12),     # Middle
    (9, 13), (13, 14), (14, 15), (15, 16),   # Ring
    (13, 17), (17, 18), (18, 19), (19, 20),  # Pinky
    (0, 17)                                   # Palm base
]


def _draw_hand_landmarks(frame_bgr, hand_landmarks, handedness_label=None):
    h, w = frame_bgr.shape[:2]

    # Draw connections
    pts = []
    for lm in hand_landmarks:
        x, y = int(lm.x * w), int(lm.y * h)
        pts.append((x, y))

    for a, b in HAND_CONNECTIONS:
        if a < len(pts) and b < len(pts):
            cv2.line(frame_bgr, pts[a], pts[b], (255, 255, 255), 2)

    # Draw points
    for (x, y) in pts:
        cv2.circle(frame_bgr, (x, y), 3, (255, 255, 255), -1)

    # Optional label near wrist (landmark 0)
    if handedness_label and pts:
        x0, y0 = pts[0]
        cv2.putText(
            frame_bgr,
            handedness_label,
            (x0 + 10, y0 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )


def _draw_face_detections(frame_bgr, detections):
    h, w = frame_bgr.shape[:2]

    for det in detections:
        # Bounding box is in pixel coordinates (origin_x/y/width/height)
        bbox = det.bounding_box
        x1 = int(bbox.origin_x)
        y1 = int(bbox.origin_y)
        x2 = int(bbox.origin_x + bbox.width)
        y2 = int(bbox.origin_y + bbox.height)

        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # Score (if present)
        score = None
        if getattr(det, "categories", None) and len(det.categories) > 0:
            score = det.categories[0].score

        if score is not None:
            cv2.putText(
                frame_bgr,
                f"face {score:.2f}",
                (x1, max(0, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 0, 0),
                2,
                cv2.LINE_AA,
            )

        # Keypoints are normalized -> scale to pixels
        if getattr(det, "keypoints", None):
            for kp in det.keypoints:
                if kp.x is None or kp.y is None:
                    continue
                cx, cy = int(kp.x * w), int(kp.y * h)
                cv2.circle(frame_bgr, (cx, cy), 3, (255, 0, 0), -1)

def main():
    if not HAND_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Missing hand model: {HAND_MODEL_PATH}\n"
            "Put your hand_landmarker.task into the models/ folder."
        )

    if not FACE_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Missing face model: {FACE_MODEL_PATH}\n"
            "Download blaze_face_short_range.tflite into models/ (see instructions)."
        )

    # MediaPipe Tasks setup
    BaseOptions = mp.tasks.BaseOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions

    FaceDetector = mp.tasks.vision.FaceDetector
    FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions

    hand_options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(HAND_MODEL_PATH)),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    # FaceDetector supports creation from options; the model here is a .tflite file.
    face_options = FaceDetectorOptions(
        base_options=BaseOptions(model_asset_path=str(FACE_MODEL_PATH)),
        running_mode=VisionRunningMode.VIDEO,
        min_detection_confidence=0.5,
        min_suppression_threshold=0.3,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam (VideoCapture(0)).")

    # Try to request a sane size (optional; camera may ignore)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("Hand + Face (MediaPipe Tasks) running... Press ESC or Q to quit.")

    start = time.perf_counter()

    with HandLandmarker.create_from_options(hand_options) as hand_landmarker, \
         FaceDetector.create_from_options(face_options) as face_detector:

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # Mirror for a more natural webcam UX
            frame = cv2.flip(frame, 1)

            # Convert BGR -> RGB for MediaPipe
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            # Monotonic timestamp (required for VIDEO mode)
            timestamp_ms = int((time.perf_counter() - start) * 1000)

            hand_result = hand_landmarker.detect_for_video(mp_image, timestamp_ms)
            face_result = face_detector.detect_for_video(mp_image, timestamp_ms)

            # Draw hands
            if hand_result and hand_result.hand_landmarks:
                for i, hand_lms in enumerate(hand_result.hand_landmarks):
                    label = None
                    if hand_result.handedness and i < len(hand_result.handedness):
                        # handedness[i] is a list of Category objects (take top-1)
                        if hand_result.handedness[i]:
                            label = hand_result.handedness[i][0].category_name
                            if label == "Left":
                                label = "Right"
                            elif label == "Right":
                                label = "Left"
                    _draw_hand_landmarks(frame, hand_lms, label)

            # Draw faces
            if face_result and getattr(face_result, "detections", None):
                _draw_face_detections(frame, face_result.detections)

            cv2.imshow("hand_face_detect", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key in (ord("q"), ord("Q")):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
