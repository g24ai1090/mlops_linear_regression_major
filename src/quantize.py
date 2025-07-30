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

    coef_min = coef.copy()
    coef_max = coef.copy()
    coef_scale = np.ones_like(coef)

    coef_quant = np.zeros_like(coef, dtype=np.uint8)
    for i in range(len(coef)):
        coef_min[i] = coef[i]
        coef_max[i] = coef[i]
        coef_scale[i] = 1.0 if coef_max[i] == coef_min[i] else 255.0 / (coef_max[i] - coef_min[i])
        coef_quant[i] = np.round((coef[i] - coef_min[i]) * coef_scale[i]).astype(np.uint8)

    intercept_quant = np.round(intercept).astype(np.int16)

    save_params({
        "coef_quant": coef_quant,
        "coef_min": coef_min,
        "coef_scale": coef_scale,
        "intercept_quant": intercept_quant
    }, QUANT_PATH)

    print("Model parameters quantized successfully")

def test_dequantized_inference():
    qparams = load_params(QUANT_PATH)
    coef_q = qparams["coef_quant"]
    coef_min = qparams["coef_min"]
    coef_scale = qparams["coef_scale"]

    coef_dq = np.zeros_like(coef_q, dtype=np.float32)
    for i in range(len(coef_q)):
        coef_dq[i] = coef_q[i] / coef_scale[i] + coef_min[i]

    print("Dequantized coefficients (sample):", coef_dq[:5])
    return coef_dq

def print_comparison_table(coef_dq):
    model = load_model(MODEL_PATH)
    _, X_test, _, y_test = load_data()

    y_pred_unquant = model.predict(X_test)
    r2_unquant = r2_score(y_test, y_pred_unquant)
    mse_unquant = mean_squared_error(y_test, y_pred_unquant)
    unquant_size = os.path.getsize(MODEL_PATH) / 1024

    qparams = load_params(QUANT_PATH)
    intercept_q = qparams["intercept_quant"]
    intercept_dq = intercept_q.astype(np.float32)

    y_pred_quant = np.dot(X_test, coef_dq) + intercept_dq
    r2_quant = r2_score(y_test, y_pred_quant)
    mse_quant = mean_squared_error(y_test, y_pred_quant)
    quant_size = os.path.getsize(QUANT_PATH) / 1024

    print("\nModel Comparison Table")
    print("| Model Version   | File Size (KB) | R² Score | Mean Squared Error |")
    print("|-----------------|----------------|----------|--------------------|")
    print(f"| Unquantized     | {unquant_size:.2f} KB   | {r2_unquant:.4f}   | {mse_unquant:.4f} |")
    print(f"| Quantized       | {quant_size:.2f} KB   | {r2_quant:.4f}   | {mse_quant:.4f} |")

if __name__ == "__main__":
    quantize_model()
    coef_dq = test_dequantized_inference()
    print_comparison_table(coef_dq)
