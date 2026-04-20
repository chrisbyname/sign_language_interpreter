import cv2  #webcame and drawing
import time #timestamp for mediapipe
import csv #saving data to csv
import os #saving data to file
import mediapipe as mp #hand detection AI model
import numpy as np #math on landmark arrays
from pathlib import Path #saving files

#dictionary of labels and their corresponding hand signs
LABELS = [
    "ok", "stop", "thumbs_up", "thumbs_down", "peace",
    "fist", "open_palm", "point", "call_me", "rock",
    "pinch", "love_you"
]

#paths for model and data storage
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "hand_landmarker.task"
DATA_DIR = BASE_DIR / "data"
CSV_PATH = DATA_DIR / "words.csv"

#hand skeleton connections for drawing
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]

#creates the data directory and csv file if they don't exist, and writes the header row to the csv
def ensure_csv():
    DATA_DIR.mkdir(exist_ok=True)
    if not CSV_PATH.exists():
        with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            header = ["label"] + [f"f{i}" for i in range(63)]  # 21 landmarks * 3 cordinates
            writer.writerow(header)

#draws the hand landmarks and connections on the video frame for visualization
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

#converts the 21 hand landmarks into a normalized 63-length vector for machine learning (claude code)
def normalize_landmarks(landmarks):
    pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)
    wrist = pts[0].copy()
    pts = pts - wrist
    scale = np.linalg.norm(pts[9])  # middle_mcp (rough hand size)
    if scale > 1e-6:
        pts = pts / scale
    return pts.flatten()

#main function to run the data collection loop, capture webcam frames, detect hand landmarks, and save labeled data to csv
def main():
    #ensures the hand landmark model exists before starting, and sets up the csv file for data storage
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at: {MODEL_PATH}")

    ensure_csv()

    #options for the hand landmarker
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

    #opens the webcam for video capture, and checks if it was successful
    cap = cv2.VideoCapture(0) #0 for default webcam
    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        return

    #initializes the label index and saved count
    label_idx = 0
    saved = 0

    #prints instructions for the user in the console
    print("Collecting data...")
    print("N=next label | B=prev label | R=record | ESC=quit")


    #main loop
    with HandLandmarker.create_from_options(options) as landmarker:
        while True: 
            ret, frame = cap.read() #capture a frame from the webcam
            if not ret:
                break

            frame = cv2.flip(frame, 1) #mirror the frame for a more intuitive experience
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #opencv uses BGR, mediapipe uses RGB, so we convert the color space
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            timestamp_ms = int(time.time() * 1000) #timestamp is required in VIDEO mode for mediapipe
            result = landmarker.detect_for_video(mp_image, timestamp_ms) #runs the hand detection model on the current frame

            current_label = LABELS[label_idx] #sets the current gesture label based on the index
            features = None 
            handedness_text = ""

            #if a hand is detected, draw the landmarks and connections
            if result.hand_landmarks:
                hand = result.hand_landmarks[0]
                draw_hand(frame, hand) 

                features = normalize_landmarks(hand)

                #if the model also provides handedness (left/right), we display that as well
                if result.handedness and result.handedness[0]:
                    handedness_text = result.handedness[0][0].category_name

            #ui overlay
            cv2.putText(frame, f"Label: {current_label}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 3)

            cv2.putText(frame, f"Hand: {handedness_text}", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            cv2.putText(frame, f"Saved: {saved}", (20, 115),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            cv2.putText(frame, "R=record | N=next | B=back | ESC=quit", (20, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.imshow("Collect Words", frame)

            #check for any key presses
            key = cv2.waitKey(1) & 0xFF

            if key == 27:  #escape key to quit
                break

            elif key in [ord("n"), ord("N")]: #n key for next label
                label_idx = (label_idx + 1) % len(LABELS)

            elif key in [ord("b"), ord("B")]: #b key for previous label
                label_idx = (label_idx - 1) % len(LABELS)

            elif key in [ord("r"), ord("R")]: #r key to record the current hand pose with the current label
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
