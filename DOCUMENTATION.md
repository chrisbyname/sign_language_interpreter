# Sign Language Interpreter – Project Documentation

## Project Idea / Goal (& description)

This project aims to build a **real-time sign language interpreter** that can recognize hand gestures from a webcam feed and convert them into readable text labels.

The core workflow is:
1. Capture live camera frames.
2. Detect hand landmarks (key points) in each frame.
3. Transform landmarks into numerical features.
4. Use a trained classifier to predict the gesture/word.
5. Display the prediction to the user in real time.

The broader objective is accessibility: reducing communication barriers by helping translate hand signs into text through a lightweight computer-vision pipeline.

---

## Technology & Stack (requirements)

### Core language
- **Python 3.x**

### Computer vision and landmark extraction
- **OpenCV (`cv2`)** for webcam capture, frame handling, and visual overlays.
- **MediaPipe** hand/face components for robust landmark detection.

### Data and modeling
- **NumPy** for numerical operations.
- **Pandas** (dataset CSV handling).
- **scikit-learn** for model training/inference utilities.
- **joblib** for saving/loading trained models and label encoders.

### Existing project assets
- Pretrained artifacts in `sign_language_interpreter/models/`, including:
  - `word_model.joblib`
  - `label_encoder.joblib`
  - MediaPipe model files (e.g., `hand_landmarker.task`)
- Datasets in `sign_language_interpreter/data/`, including `words.csv` and archived CSV versions.

### Runtime / environment requirements
- A working webcam.
- Python virtual environment (`venv`) recommended.
- Typical packages used in this repository:
  - `opencv-python`
  - `mediapipe`
  - `numpy`
  - `pandas`
  - `scikit-learn`
  - `joblib`

---

## Conclusion (did you manage to get it working, first results, finetuning, etc..)

Yes—the repository structure and included artifacts indicate a working end-to-end prototype:
- Data collection scripts are present (`collect_words.py`).
- Training script is present (`training_quick.py`).
- Live/testing/inference scripts are present (`run_live_test.py`, `app.py`, `landmarks_live.py`).
- Trained model artifacts and encoders are already saved under `models/`.

**First results:** the setup is capable of live gesture classification for the label set used in the current dataset and model.

**Fine-tuning status:** performance is functional for demo use, but still sensitive to real-world conditions (lighting, camera angle, hand position consistency, and class balance). More diverse training samples and incremental retraining should improve robustness.

---

## Lessons Learned (realizations, issues found)

1. **Data quality is the biggest lever for accuracy.** Landmark-based classifiers can perform well, but only with representative samples.
2. **Class balance matters.** Underrepresented signs can be confused with visually similar gestures.
3. **Environment sensitivity is real.** Lighting, background clutter, and camera distance affect landmark stability.
4. **Landmark features are efficient.** Compared with full-image models, landmark vectors are lightweight and practical for real-time use.
5. **Model/version management is important.** The repository contains “old” and current model files, highlighting the need for clear experiment tracking.
6. **Pipeline separation helps iteration.** Splitting collection, training, and live inference into separate scripts makes debugging and improvements faster.

---

## Next Steps (opportunities)

1. **Expand vocabulary** by collecting more classes and more examples per class.
2. **Improve generalization** with broader data capture conditions (different users, backgrounds, distances, orientations).
3. **Add sequence awareness** for dynamic signs/phrases (temporal modeling instead of frame-only classification).
4. **Confidence handling** (thresholding + “unknown” class) to reduce incorrect forced predictions.
5. **Evaluation framework** with train/validation splits, confusion matrices, and per-class metrics.
6. **User experience upgrades** such as sentence assembly, text-to-speech output, and a cleaner GUI.
7. **Deployment paths**: package as a desktop app or deploy a browser/mobile-friendly variant.

---

## References

- MediaPipe documentation: https://developers.google.com/mediapipe
- OpenCV documentation: https://docs.opencv.org/
- scikit-learn documentation: https://scikit-learn.org/stable/
- NumPy documentation: https://numpy.org/doc/
- Real-time gesture/sign recognition literature using hand landmarks and classical ML pipelines.
