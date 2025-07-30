import os
import numpy as np
from src.utils import load_model, save_params, load_params

MODEL_PATH = "models/linear_model.joblib"
UNQUANT_PATH = "models/unquant_params.joblib"
QUANT_PATH = "models/quant_params.joblib"

def quantize():
    model = load_model(MODEL_PATH)
    coef = model.coef_
    intercept = model.intercept_

    # Save original parameters
    save_params({"coef": coef, "intercept": intercept}, UNQUANT_PATH)

    # Quantization (scale to uint8)
    coef_min, coef_max = coef.min(), coef.max()
    scale = 255.0 / (coef_max - coef_min)
    coef_quant = ((coef - coef_min) * scale).astype(np.uint8)
    intercept_quant = np.array([intercept], dtype=np.float32)  # keep as float for simplicity

    save_params({
        "coef_quant": coef_quant,
        "intercept_quant": intercept_quant,
        "coef_min": coef_min,
        "coef_max": coef_max,
        "scale": scale
    }, QUANT_PATH)

    print(" Model parameters quantized successfully")

def test_dequantized_inference():
    """Test inference using dequantized coefficients."""
    qparams = load_params(QUANT_PATH)
    coef_quant = qparams["coef_quant"]
    coef_min = qparams["coef_min"]
    coef_max = qparams["coef_max"]
    scale = qparams["scale"]

    # Dequantize
    coef_dequant = coef_quant.astype(np.float32) / scale + coef_min

    print("Dequantized coefficients (sample):", coef_dequant[:5])

if __name__ == "__main__":
    quantize()
    test_dequantized_inference()
