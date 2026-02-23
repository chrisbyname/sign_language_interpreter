import time
import random
from pathlib import Path
from collections import deque, Counter

import cv2
import numpy as np
import mediapipe as mp
import joblib

import tkinter as tk
from PIL import Image, ImageTk


# ---------------- Paths ----------------
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
ASSETS_DIR = BASE_DIR / "assets" / "signs"

TASK_MODEL_PATH = MODELS_DIR / "hand_landmarker.task"
MODEL_PATH = MODELS_DIR / "word_model.joblib"
ENCODER_PATH = MODELS_DIR / "label_encoder.joblib"


# ---------------- Hand skeleton connections ----------------
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]


def most_common(items):
    if not items:
        return "..."
    return Counter(items).most_common(1)[0][0]


def normalize_landmarks(landmarks):
    """
    21 landmarks -> 63 features
    - subtract wrist (translation invariance)
    - divide by wrist->middle_mcp distance (scale invariance)
    """
    pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)

    wrist = pts[0].copy()
    pts = pts - wrist

    scale = np.linalg.norm(pts[9])  # wrist -> middle mcp
    if scale > 1e-6:
        pts = pts / scale

    return pts.flatten().reshape(1, -1)


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


def overlay_rgba(frame_bgr, overlay_rgba, x, y):
    """
    Alpha-blend overlay RGBA image onto BGR frame at top-left (x,y).
    Clips to frame bounds.
    """
    h, w = overlay_rgba.shape[:2]
    H, W = frame_bgr.shape[:2]

    if x >= W or y >= H:
        return

    # Clip overlay region
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(W, x + w), min(H, y + h)

    overlay_x1 = x1 - x
    overlay_y1 = y1 - y
    overlay_x2 = overlay_x1 + (x2 - x1)
    overlay_y2 = overlay_y1 + (y2 - y1)

    roi = frame_bgr[y1:y2, x1:x2]
    ov = overlay_rgba[overlay_y1:overlay_y2, overlay_x1:overlay_x2]

    alpha = ov[:, :, 3].astype(np.float32) / 255.0
    alpha = alpha[..., None]

    roi[:] = (1.0 - alpha) * roi + alpha * ov[:, :, :3]
    frame_bgr[y1:y2, x1:x2] = roi


class SimonSignsApp:
    def __init__(self):
        # --- Load model ---
        if not TASK_MODEL_PATH.exists():
            raise FileNotFoundError(f"Missing: {TASK_MODEL_PATH}")
        if not MODEL_PATH.exists() or not ENCODER_PATH.exists():
            raise FileNotFoundError(
                "Missing trained model files. Make sure these exist:\n"
                f"- {MODEL_PATH}\n- {ENCODER_PATH}"
            )

        self.model = joblib.load(MODEL_PATH)
        self.le = joblib.load(ENCODER_PATH)

        self.labels = list(self.le.classes_)

        # If you ever add "none", exclude it from the game targets
        self.play_labels = [lbl for lbl in self.labels if lbl.lower() not in ("none", "idle")]

        if len(self.play_labels) < 2:
            raise ValueError("You need at least 2 playable labels for the game.")

        # --- Load sign images into memory (as RGBA numpy arrays) ---
        self.sign_imgs = {}
        for lbl in self.play_labels:
            p = ASSETS_DIR / f"{lbl}.png"
            if p.exists():
                pil = Image.open(p).convert("RGBA")
                pil = pil.resize((160, 160), Image.LANCZOS)
                self.sign_imgs[lbl] = np.array(pil, dtype=np.uint8)
            else:
                # placeholder (grey)
                ph = Image.new("RGBA", (160, 160), (180, 180, 180, 255))
                self.sign_imgs[lbl] = np.array(ph, dtype=np.uint8)

        # --- MediaPipe Tasks ---
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

        self.landmarker = HandLandmarker.create_from_options(options)

        # --- Webcam ---
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Could not open webcam.")

        # --- Prediction smoothing ---
        self.history = deque(maxlen=10)
        self.last_pred = "..."
        self.last_conf = 0.0

        # --- Game state ---
        self.game_active = False
        self.score = 0

        self.round_duration = 5.0
        self.round_start = None
        self.target_label = None

        self.hand_last_seen = 0.0
        self.stop_if_no_hand_seconds = 0.8

        # How strict the "correct" detection should be
        self.conf_threshold = 0.60
        self.required_stable_frames = 6

        # --- Tkinter Window ---
        self.root = tk.Tk()
        self.root.title("Simon Says: Sign Edition")
        self.root.geometry("1100x700")
        self.root.configure(bg="#111111")

        self.video_label = tk.Label(self.root, bg="#111111")
        self.video_label.pack(padx=10, pady=10, fill="both", expand=True)

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        # Start loop
        self.update_frame()

    def pick_new_target(self):
        self.target_label = random.choice(self.play_labels)
        self.round_start = time.time()

    def start_game(self):
        self.game_active = True
        self.score = 0
        self.history.clear()
        self.pick_new_target()

    def stop_game(self):
        self.game_active = False
        self.round_start = None
        self.target_label = None
        self.history.clear()

    def is_stable_correct(self):
        # stable majority check
        if len(self.history) < self.required_stable_frames:
            return False
        recent = list(self.history)[-self.required_stable_frames:]
        stable = all(x == recent[0] for x in recent)

        if not stable:
            return False

        return recent[0] == self.target_label and self.last_conf >= self.conf_threshold

    def draw_timer_bar(self, frame, remaining_ratio):
        # top bar background
        h, w, _ = frame.shape
        bar_h = 16
        cv2.rectangle(frame, (0, 0), (w, bar_h), (50, 50, 50), -1)

        # progress bar
        fill_w = int(w * max(0.0, min(1.0, remaining_ratio)))
        cv2.rectangle(frame, (0, 0), (fill_w, bar_h), (255, 255, 255), -1)

    def draw_center_target(self, frame):
        if not self.target_label:
            return

        h, w, _ = frame.shape

        # Big label text
        text = f"DO: {self.target_label.upper()}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 1.4
        thickness = 4

        (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)

        # Put text slightly above center
        text_x = (w - tw) // 2
        text_y = int(h * 0.30)

        cv2.putText(frame, text, (text_x, text_y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)

        # Overlay icon in the middle
        #icon = self.sign_imgs.get(self.target_label)
        #if icon is not None:
        #    icon_h, icon_w = icon.shape[:2]
        #    icon_x = (w - icon_w) // 2
        #    icon_y = int(h * 0.35)
        #    overlay_rgba(frame, icon, icon_x, icon_y)

    def draw_score(self, frame):
        h, w, _ = frame.shape
        txt = f"Score: {self.score}"
        cv2.putText(frame, txt, (w - 260, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 3, cv2.LINE_AA)

    def draw_wait_screen(self, frame):
        h, w, _ = frame.shape
        msg = "Show your hand to start"
        (tw, th), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
        x = (w - tw) // 2
        y = (h + th) // 2
        cv2.putText(frame, msg, (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3, cv2.LINE_AA)

        if self.score > 0:
            msg2 = f"Last score: {self.score}"
            (tw2, th2), _ = cv2.getTextSize(msg2, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
            x2 = (w - tw2) // 2
            y2 = y + 50
            cv2.putText(frame, msg2, (x2, y2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.root.after(30, self.update_frame)
            return

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        timestamp_ms = int(time.time() * 1000)
        result = self.landmarker.detect_for_video(mp_image, timestamp_ms)

        now = time.time()
        hand_present = bool(result.hand_landmarks)

        # Hand presence management
        if hand_present:
            self.hand_last_seen = now
        else:
            # stop game if hand missing for some time
            if self.game_active and (now - self.hand_last_seen) > self.stop_if_no_hand_seconds:
                self.stop_game()

        # If hand appears and game not active -> auto start
        if hand_present and not self.game_active:
            self.start_game()

        # Prediction
        self.last_pred = "..."
        self.last_conf = 0.0

        if hand_present:
            hand = result.hand_landmarks[0]
            draw_hand(frame, hand)

            X = normalize_landmarks(hand)
            proba = self.model.predict_proba(X)[0]
            pred_idx = int(np.argmax(proba))
            self.last_conf = float(proba[pred_idx])
            pred_label = self.le.inverse_transform([pred_idx])[0]

            # Apply confidence threshold into smoothing history
            if self.last_conf >= 0.50:
                self.history.append(pred_label)
            else:
                self.history.append("...")

            self.last_pred = most_common(self.history)

        # Game round logic
        if self.game_active and self.target_label and self.round_start is not None:
            elapsed = now - self.round_start
            remaining = max(0.0, self.round_duration - elapsed)
            remaining_ratio = remaining / self.round_duration

            # Timer bar
            self.draw_timer_bar(frame, remaining_ratio)

            # Target display
            self.draw_center_target(frame)

            # Score
            self.draw_score(frame)

            # Check win
            if self.is_stable_correct():
                self.score += 1
                self.pick_new_target()
                self.history.clear()

            # If time expired -> new round (no points)
            elif elapsed >= self.round_duration:
                self.pick_new_target()
                self.history.clear()

            # Also show current predicted label small (top-left)
            cv2.putText(frame, f"Pred: {self.last_pred}", (20, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3, cv2.LINE_AA)
            cv2.putText(frame, f"Conf: {self.last_conf:.2f}", (20, 85),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        else:
            # waiting screen
            self.draw_wait_screen(frame)

        # --- Convert to Tkinter image WITHOUT stretching ---
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)

        max_w = self.video_label.winfo_width()
        max_h = self.video_label.winfo_height()
        if max_w < 50 or max_h < 50:
            max_w, max_h = 1050, 600

        orig_w, orig_h = pil_img.size
        scale = min(max_w / orig_w, max_h / orig_h)
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)

        pil_img = pil_img.resize((new_w, new_h), Image.LANCZOS)

        imgtk = ImageTk.PhotoImage(image=pil_img)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

        self.root.after(20, self.update_frame)

    def on_close(self):
        try:
            self.cap.release()
        except:
            pass
        try:
            self.landmarker.close()
        except:
            pass
        self.root.destroy()

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    app = SimonSignsApp()
    app.run()
