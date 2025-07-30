import joblib
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

def load_data(test_size=0.2, random_state=42):
    """Load the California Housing dataset and split into train/test."""
    data = fetch_california_housing(as_frame=True)
    X = data.data
    y = data.target
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def save_model(model, path):
    """Save a trained model to disk."""
    joblib.dump(model, path)

def load_model(path):
    """Load a trained model from disk."""
    return joblib.load(path)

def save_params(params, path):
    """Save parameters (coefficients & intercept) to disk."""
    joblib.dump(params, path)

def load_params(path):
    """Load parameters (coefficients & intercept) from disk."""
    return joblib.load(path)
