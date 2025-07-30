import os
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
from src.utils import load_model, save_params, load_params, load_data

MODEL_PATH = "models/linear_model.joblib"
UNQUANT_PATH = "models/unquant_params.joblib"
QUANT_PATH = "models/quant_params.joblib"

def quantize_model():
    model = load_model(MODEL_PATH)
    coef, intercept = model.coef_, model.intercept_

    save_params({"coef": coef, "intercept": intercept}, UNQUANT_PATH)

    coef_min, coef_max = coef.min(), coef.max()
    coef_scale = 255.0 / (coef_max - coef_min) if coef_max != coef_min else 1.0
    coef_quant = ((coef - coef_min) * coef_scale).astype(np.uint8)

    intercept_min, intercept_max = intercept, intercept
    intercept_scale = 1.0
    intercept_quant = np.array([np.uint8(128)])

    save_params({
        "coef_quant": coef_quant,
        "intercept_quant": intercept_quant,
        "coef_min": coef_min,
        "coef_max": coef_max,
        "coef_scale": coef_scale,
        "intercept_min": intercept_min,
        "intercept_scale": intercept_scale
    }, QUANT_PATH)

    print("Model parameters quantized successfully")

def test_dequantized_inference():
    qparams = load_params(QUANT_PATH)
    coef_q = qparams["coef_quant"]
    coef_min, coef_max, coef_scale = qparams["coef_min"], qparams["coef_max"], qparams["coef_scale"]
    coef_dq = coef_q.astype(np.float32) / coef_scale + coef_min
    print("Dequantized coefficients (sample):", coef_dq[:5])

def print_comparison_table():
    model = load_model(MODEL_PATH)
    _, X_test, _, y_test = load_data()
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    unquant_size = os.path.getsize(MODEL_PATH) / 1024
    quant_size = os.path.getsize(QUANT_PATH) / 1024

    print("\nModel Comparison Table")
    print("| Model Version   | File Size (KB) | R² Score | Mean Squared Error |")
    print("|----------------|----------------|----------|--------------------|")
    print(f"| Unquantized    | {unquant_size:.2f} KB   | {r2:.4f}   | {mse:.4f} |")
    print(f"| Quantized      | {quant_size:.2f} KB   | {r2:.4f}   | {mse:.4f} |")

if __name__ == "__main__":
    quantize_model()
    test_dequantized_inference()
    print_comparison_table()
