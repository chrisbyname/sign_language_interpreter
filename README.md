# Sign Language Interpreter

## Project Idea / Goal

The goal of this project is to build a **real-time sign language interpreter** using computer vision and machine learning. The system captures hand gestures through a webcam, extracts structured hand landmark coordinates, and classifies them into recognizable sign language letters or gestures.

This project demonstrates how artificial intelligence and computer vision can be applied to accessibility challenges by translating visual sign input into readable text output.

### Core Objectives

- Detect hands in real-time using a webcam  
- Extract 21 hand landmark coordinates per detected hand  
- Train a machine learning model on landmark data  
- Predict and display sign language gestures live  

---

## How It Works

1. The webcam captures live video input.
2. MediaPipe detects hand landmarks (21 key points per hand).
3. Landmark coordinates are converted into a structured numerical feature vector.
4. A trained model predicts the corresponding sign.
5. The predicted label is displayed on the screen in real time.

---

## Technology & Stack

### Programming Language
- Python 3.x

### Computer Vision
- OpenCV – video capture and frame processing  
- MediaPipe – hand landmark detection  

### Machine Learning
- TensorFlow / Keras (or chosen ML framework)  
- NumPy – numerical data handling  
- Scikit-learn (for preprocessing and evaluation if used)  

### Environment
- Virtual environment recommended (venv)

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/chrisbyname/sign_language_interpreter.git
cd sign_language_interpreter
