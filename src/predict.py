import numpy as np
from src.utils import load_model, load_data

MODEL_PATH = "models/linear_model.joblib"

def run_prediction():
    model = load_model(MODEL_PATH)
    _, X_test, _, _ = load_data()

    preds = model.predict(X_test[:5]) 
    print("Predictions on test data:")
    for i, p in enumerate(preds, start=1):
        print(f"Sample {i}: {p:.4f}")

if __name__ == "__main__":
    run_prediction()
