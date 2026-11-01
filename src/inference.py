"""Inferencia para un trayecto individual (usada por la app web)."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C  # noqa: E402
from src.models import RNNForecaster  # noqa: E402
from src.train import predict  # noqa: E402
from src.windows import Scaler, build_arrays, make_anchors  # noqa: E402


def load_bundle(device="cpu"):
    scaler = Scaler.from_json(C.MODEL_DIR / "scaler.json")
    meta_p = C.MODEL_DIR / "model_meta.json"
    residual = json.loads(meta_p.read_text()).get("residual", False) if meta_p.exists() else False
    models = {}
    for cell in ["lstm", "gru"]:
        p = C.MODEL_DIR / f"{cell}.pt"
        if p.exists():
            m = RNNForecaster(cell)
            m.load_state_dict(torch.load(p, map_location=device))
            models[cell] = m.eval()
    return models, scaler, residual


def forecast_trip(bins: pd.DataFrame, models, scaler, residual, hist_rate: float) -> pd.DataFrame:
    """bins: un trayecto con columnas step, speed, rpm, load, fcr (pasos de STEP_S s).
    Devuelve un pronóstico por bloque de HORIZON pasos para cada método."""
    b = bins.copy()
    b["veh"], b["trip"] = 0, 0
    arr = build_arrays(b, scaler)
    starts, target, _, ts, blen, pers = make_anchors(arr, C.HORIZON, False, scaler.y_mean)
    out = pd.DataFrame({"t": ts, "block_len": blen, "real": target})
    X = torch.as_tensor(arr.X)
    for cell, m in models.items():
        out[cell] = predict(m, X, starts, scaler, torch.device("cpu"), base=pers if residual else None)
    out["hist"] = hist_rate
    out["persist"] = np.where(ts > 0, pers, hist_rate)
    hrs = out["block_len"] * C.STEP_S / 3600.0
    for c in ["real", "lstm", "gru", "hist", "persist"]:
        if c in out:
            out[f"L_{c}"] = out[c] * hrs
    return out
