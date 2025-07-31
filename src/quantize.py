import os
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from src.utils import load_model, save_params, load_params, load_data

MODEL_PATH = "models/linear_model.joblib"
UNQUANT_PATH = "models/unquant_params.joblib"
QUANT_PATH = "models/quant_params.joblib"
QUANT16_PATH = "models/quant16_params.joblib"

def quantize_model():
    model = load_model(MODEL_PATH)
    coef, intercept = model.coef_, model.intercept_

    save_params({"coef": coef, "intercept": intercept}, UNQUANT_PATH)

    coef_min_8 = np.zeros_like(coef)
    coef_max_8 = np.zeros_like(coef)
    coef_scale_8 = np.zeros_like(coef)
    coef_quant_8 = np.zeros_like(coef, dtype=np.uint8)

    for i in range(len(coef)):
        coef_min_8[i] = coef[i]
        coef_max_8[i] = coef[i]
        coef_scale_8[i] = 1.0 if coef_max_8[i] == coef_min_8[i] else 255.0 / (coef_max_8[i] - coef_min_8[i])
        coef_quant_8[i] = np.round((coef[i] - coef_min_8[i]) * coef_scale_8[i]).astype(np.uint8)

    intercept_quant_8 = np.round(intercept).astype(np.int16)

    save_params({
        "coef_quant": coef_quant_8,
        "coef_min": coef_min_8,
        "coef_scale": coef_scale_8,
        "intercept_quant": intercept_quant_8
    }, QUANT_PATH)

    coef_min_16 = coef.min()
    coef_max_16 = coef.max()
    coef_scale_16 = 65535.0 / (coef_max_16 - coef_min_16)
    coef_quant_16 = ((coef - coef_min_16) * coef_scale_16).astype(np.uint16)

    intercept_quant_16 = np.round(intercept).astype(np.int32)

    save_params({
        "coef_quant16": coef_quant_16,
        "coef_min16": coef_min_16,
        "coef_max16": coef_max_16,
        "coef_scale16": coef_scale_16,
        "intercept_quant16": intercept_quant_16
    }, QUANT16_PATH)

    print("Model parameters quantized successfully (8-bit per-coefficient & 16-bit global)")

def dequantize(bits=8):
    if bits == 8:
        qparams = load_params(QUANT_PATH)
        coef_q = qparams["coef_quant"]
        coef_min = qparams["coef_min"]
        coef_scale = qparams["coef_scale"]

        coef_dq = np.zeros_like(coef_q, dtype=np.float32)
        for i in range(len(coef_q)):
            coef_dq[i] = coef_q[i] / coef_scale[i] + coef_min[i]

        intercept_dq = qparams["intercept_quant"].astype(np.float32)
        return coef_dq, intercept_dq

    elif bits == 16:
        qparams = load_params(QUANT16_PATH)
        coef_q = qparams["coef_quant16"]
        coef_min = qparams["coef_min16"]
        coef_scale = qparams["coef_scale16"]

        coef_dq = coef_q.astype(np.float32) / coef_scale + coef_min
        intercept_dq = qparams["intercept_quant16"].astype(np.float32)
        return coef_dq, intercept_dq

    else:
        raise ValueError("Only 8 or 16 bits supported")

def evaluate_model(X_test, y_test, coef, intercept, label):
    y_pred = np.dot(X_test, coef) + intercept
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    return {
        "label": label,
        "r2": r2,
        "mse": mse,
        "mae": mae,
        "y_pred_sample": y_pred[:5]
    }

def print_comparison_table():
    model = load_model(MODEL_PATH)
    _, X_test, _, y_test = load_data()

    results = []

    coef, intercept = model.coef_, model.intercept_
    results.append(evaluate_model(X_test, y_test, coef, intercept, "Sklearn (Unquantized)"))

    qparams8 = load_params(QUANT_PATH)
    coef_q8 = qparams8["coef_quant"].astype(np.float32)
    intercept_q8 = qparams8["intercept_quant"].astype(np.float32)
    results.append(evaluate_model(X_test, y_test, coef_q8, intercept_q8, "8-bit Quantized"))

    coef_dq8, intercept_dq8 = dequantize(bits=8)
    results.append(evaluate_model(X_test, y_test, coef_dq8, intercept_dq8, "8-bit Dequantized"))

    qparams16 = load_params(QUANT16_PATH)
    coef_q16 = qparams16["coef_quant16"].astype(np.float32)
    intercept_q16 = qparams16["intercept_quant16"].astype(np.float32)
    results.append(evaluate_model(X_test, y_test, coef_q16, intercept_q16, "16-bit Quantized"))

    coef_dq16, intercept_dq16 = dequantize(bits=16)
    results.append(evaluate_model(X_test, y_test, coef_dq16, intercept_dq16, "16-bit Dequantized"))

    print("\n Full Model Comparison Table")
    print("| Version              | R² Score | MSE       | MAE       | Sample Predictions |")
    print("|----------------------|----------|-----------|-----------|-------------------|")
    for res in results:
        print(f"| {res['label']:<20} | {res['r2']:.4f}   | {res['mse']:.4f} | {res['mae']:.4f} | {np.round(res['y_pred_sample'],2)} |")

if __name__ == "__main__":
    quantize_model()
    print_comparison_table()
