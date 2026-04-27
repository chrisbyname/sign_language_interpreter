# Sign Language Interpreter

## Project Idea / Goal

- The goal of this project is to build a real-time Sign Language Interpreter using computer vision and machine learning.
- The system captures hand gestures through a webcam.
- It extracts structured hand landmark coordinates.
- A trained model classifies each gesture.
- The predicted sign is displayed as text output.
- The project demonstrates how AI can improve accessibility and support communication.

------------------------------------------------------------------------

## How It Works

- The webcam captures live video frames.
- MediaPipe detects 21 hand landmark points.
- Landmark coordinates are transformed into numerical feature vectors.
- The trained machine learning model predicts the gesture.
- The predicted label is displayed on screen in real time.
- This loop repeats continuously for each incoming frame.

------------------------------------------------------------------------

## Technology & Stack

### Programming Language

- Python 3.14.2

### Computer Vision

- OpenCV
  - Webcam access
  - Frame processing
  - Display rendering
- MediaPipe
  - Real-time hand tracking
  - 21-landmark detection model

### Machine Learning

- Scikit-learn (RandomForestClassifier, LabelEncoder)
- NumPy
- Pandas
- Joblib

### UI & Media

- Tkinter
- Pillow (PIL)

### Development Tools

- Virtual Environment (venv)
- Git
- GitHub

------------------------------------------------------------------------

## Installation

### Clone the Repository

- git clone https://github.com/chrisbyname/sign_language_interpreter.git
- cd sign_language_interpreter

### Create a Virtual Environment

- python -m venv venv

Activate:

- Windows: `venv\Scripts\activate`
- Mac/Linux: `source venv/bin/activate`

### Install Dependencies

- pip install -r requirements.txt

------------------------------------------------------------------------

## Usage

- Run the app:
- python app.py
- The webcam window will open.
- Hand landmarks will be detected.
- Predictions will appear on screen.
- Press q to exit.

Optional supporting scripts:

- Collect training data: `python dataset_collection.py`
- Train/update model: `python training.py`

------------------------------------------------------------------------

## Project Structure

- sign_language_interpreter/
- data/ (dataset CSV files)
- models/ (saved trained models and MediaPipe task file)
- app.py (main application entry point)
- dataset_collection.py (data collection script)
- training.py (model training script)
- requirements.txt (dependencies)
- README.md (documentation)

------------------------------------------------------------------------

## Conclusion

- The project successfully detects hand landmarks in real time.
- It converts landmark data into structured model input.
- The trained model predicts sign gestures.
- Predictions are displayed live on screen.
- Text-To-Speech reads predictions aloud.
- Performance depends on lighting conditions and dataset quality.
- Fine-tuning improves accuracy and reliability.

------------------------------------------------------------------------

## Lessons Learned

- Dataset quality directly impacts model performance.
- Big datasets often lead to overfitting.
- Greater sample variation improves generalization.
- Landmark-based approaches are efficient compared to raw image training.
- Real-time performance requires optimization.
- Proper virtual environment management helps prevent dependency issues.
- A correct .gitignore prevents committing unnecessary files.
- How to correctly upload files to Github.

------------------------------------------------------------------------

## Next Steps

- Expand the gesture vocabulary.
- Add sentence-level recognition.
- Improve model accuracy with transfer learning.
- Create a graphical user interface.
- Deploy as a web or mobile application.
- Support multiple hands and gesture sequences.

------------------------------------------------------------------------

## References

- MediaPipe Documentation
- OpenCV Documentation
- Scikit-learn Documentation
- Research on Sign Language Recognition using Computer Vision

------------------------------------------------------------------------

## License

- Educational and research use only.
