import os
import pytest
from sklearn.linear_model import LinearRegression
from src.utils import load_data, load_model
from src.train import train_model

MODEL_PATH = "models/linear_model.joblib"

def test_data_loading():
    X_train, X_test, y_train, y_test = load_data()
    assert X_train.shape[0] > 0
    assert y_train.shape[0] > 0

def test_model_training_and_saving():
    model, r2 = train_model()
    assert isinstance(model, LinearRegression)
    assert hasattr(model, "coef_")
    assert os.path.exists(MODEL_PATH)
    assert r2 > 0.5  

def test_model_loading():
    model = load_model(MODEL_PATH)
    assert isinstance(model, LinearRegression)
