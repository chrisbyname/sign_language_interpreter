import time
from pathlib import Path
from collections import deque, Counter

import cv2
import numpy as np
import joblib
import mediapipe as mp

import tkinter as tk
from tkinter import ttk

from PIL import Image, ImageTk


# ---------------- Paths ----------------
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
ASSETS_DIR = BASE_DIR / "images"

TASK_MODEL_PATH = MODELS_DIR / "hand_landmarker.task"
MODEL_PATH = MODELS_DIR / "word_model.joblib"
ENCODER_PATH = MODELS_DIR / "label_encoder.joblib"


# ---------------- Hand drawing connections ----------------
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]


def most_common(items):
    return Counter(items).most_common(1)[0][0]


def normalize_landmarks(landmarks):
    """
    Convert 21 hand landmarks into a normalized feature vector (63 values).
    - subtract wrist (translation invariance)
    - divide by wrist->middle_mcp distance (scale invariance)
    """
    pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)

    wrist = pts[0].copy()
    pts = pts - wrist

    scale = np.linalg.norm(pts[9])  # wrist -> middle mcp
    if scale > 1e-6:
        pts = pts / scale

    return pts.flatten().reshape(1, -1)  # (1, 63)


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


class SignShowcaseApp:
    def __init__(self):
        # ---- Load models ----
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

        # ---- MediaPipe Tasks setup ----
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

        # ---- Webcam ----
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Could not open webcam.")

        # ---- Smoothing history ----
        self.history = deque(maxlen=10)
        self.last_pred = "..."
        self.last_conf = 0.0

        # ---- Tkinter window ----
        self.root = tk.Tk()
        self.root.title("Sign Language AI Showcase")
        self.root.geometry("1100x700")

        # Clean-ish look
        self.root.configure(bg="#FFFFFF")

        # ---- Top: video area ----
        self.video_label = tk.Label(self.root, bg="#FFFFFF")
        self.video_label.pack(padx=10, pady=10, fill="both", expand=True)

        # ---- Bottom: signs bar (scrollable) ----
        self.bottom_frame = tk.Frame(self.root, bg="#ffffff", height=180)
        self.bottom_frame.pack(fill="x", side="bottom")

        self._build_sign_bar()

        # ---- Close handler ----
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        # Start loop
        self.update_frame()

    def _build_sign_bar(self):
        # Scrollable canvas
        canvas = tk.Canvas(self.bottom_frame, bg="#ffffff", highlightthickness=0, height=180)
        canvas.pack(side="top", fill="x", expand=True)

        scrollbar = ttk.Scrollbar(self.bottom_frame, orient="horizontal", command=canvas.xview)
        scrollbar.pack(side="bottom", fill="x")

        canvas.configure(xscrollcommand=scrollbar.set)

        inner = tk.Frame(canvas, bg="#ffffff")
        canvas.create_window((0, 0), window=inner, anchor="nw")

        # Keep references so images don’t get garbage-collected
        self.icon_refs = {}
        self.icon_cards = {}

        for label in self.labels:
            card = tk.Frame(inner, bg="#ffffff", padx=10, pady=10)
            card.pack(side="left", padx=5, pady=10)

            # Load icon
            icon_path = ASSETS_DIR / f"{label}.png"
            if icon_path.exists():
                pil_img = Image.open(icon_path).convert("RGBA")
                pil_img = pil_img.resize((70, 70), Image.LANCZOS)
                tk_img = ImageTk.PhotoImage(pil_img)
            else:
                # Placeholder if missing file
                pil_img = Image.new("RGBA", (70, 70), (30, 30, 30, 255))
                tk_img = ImageTk.PhotoImage(pil_img)

            img_label = tk.Label(card, image=tk_img, bg="#ffffff")
            img_label.pack()

            text_label = tk.Label(
                card,
                text=label,
                fg="#111111",
                bg="#ffffff",
                font=("Segoe UI", 11, "bold")
            )
            text_label.pack(pady=(6, 0))

            self.icon_refs[label] = tk_img
            self.icon_cards[label] = card

        inner.update_idletasks()
        canvas.configure(scrollregion=canvas.bbox("all"))

    def _highlight_prediction(self, label):
        # Reset all cards
        for lbl, card in self.icon_cards.items():
            card.configure(bg="#ffffff")
            for w in card.winfo_children():
                w.configure(bg="#ffffff")

        # Highlight predicted one
        if label in self.icon_cards:
            card = self.icon_cards[label]
            card.configure(bg="#ffffff")
            for w in card.winfo_children():
                w.configure(bg="#ffffff")

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

        label_text = "..."
        conf = 0.0

        if result.hand_landmarks:
            hand = result.hand_landmarks[0]
            draw_hand(frame, hand)

            X = normalize_landmarks(hand)
            proba = self.model.predict_proba(X)[0]
            pred_idx = int(np.argmax(proba))
            conf = float(proba[pred_idx])
            pred_label = self.le.inverse_transform([pred_idx])[0]

            if conf > 0.55:
                self.history.append(pred_label)
            else:
                self.history.append("...")

            label_text = most_common(self.history)

        # Store
        self.last_pred = label_text
        self.last_conf = conf

        # Overlay prediction text
        cv2.putText(frame, f"Pred: {self.last_pred}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3)
        cv2.putText(frame, f"Conf: {self.last_conf:.2f}", (20, 95),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        # Highlight bottom icon
        self._highlight_prediction(self.last_pred)

        # Convert to Tkinter image
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)

        # Resize to fit nicely in window while keeping aspect


        imgtk = ImageTk.PhotoImage(image=pil_img)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

        # Next update
        self.root.after(20, self.update_frame)  # ~50 FPS target

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
    app = SignShowcaseApp()
    app.run()
