import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

#paths for model and data storage
CSV_PATH = "data/words.csv"
MODEL_PATH = "models/word_model.joblib"
ENCODER_PATH = "models/label_encoder.joblib"

#main function to load data, train model, evaluate, and save the trained model and label encoder for later use in prediction
def main():
    #Loads the CSV data file and checks for correctness
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"Could not find {CSV_PATH}. Make sure your CSV is in data/words.csv")
    df = pd.read_csv(CSV_PATH)
    if "label" not in df.columns:
        raise ValueError("CSV must contain a 'label' column.")

    #prepares the feature matrix X and label vector y, encoding string labels into numeric format for model training
    X = df.drop(columns=["label"]).values.astype(np.float32)
    y_text = df["label"].values.astype(str)

    le = LabelEncoder()
    y = le.fit_transform(y_text)

    #quick sanity print
    print("Samples per label:")
    print(df["label"].value_counts())
    print("\nLabels:", list(le.classes_))

    #split training/testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    #train model
    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced"
    )
    model.fit(X_train, y_train)

    #evaluate
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    #print evaluation stats
    print("\nAccuracy:", round(acc, 4))
    print("\nConfusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(y_test, preds))

    print("\nClassification report:")
    print(classification_report(y_test, preds, target_names=le.classes_))

    #save model and label encoder
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(le, ENCODER_PATH)

    print(f"\nSaved model -> {MODEL_PATH}")
    print(f"Saved labels -> {ENCODER_PATH}")


if __name__ == "__main__":
    main()
