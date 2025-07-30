import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from src.utils import load_data, save_model

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "linear_model.joblib")

def train_model():
    X_train, X_test, y_train, y_test = load_data()

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    print(f" Model trained successfully")
    print(f" R2 Score: {r2:.4f}")
    print(f" Mean Squared Error: {mse:.4f}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    save_model(model, MODEL_PATH)

    return model, r2

if __name__ == "__main__":
    train_model()
