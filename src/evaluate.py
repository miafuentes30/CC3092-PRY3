"""Evalúa LSTM, GRU y los métodos de referencia en validación y prueba.

Simulación del uso real: en cada trayecto se emite un pronóstico cada HORIZON pasos
(2 min) usando solo la información disponible hasta ese momento. Los litros del
trayecto se obtienen integrando los pronósticos y se comparan con el consumo real.

Métodos de referencia (sin deep learning):
- hist : promedio histórico del VEHÍCULO (L/h) calculado con sus trayectos anteriores
         (orden cronológico). Si no tiene historial: promedio del set de entrenamiento.
- persist : persistencia, consumo medio de los últimos 3 min del mismo trayecto.

Uso:
    python -m src.evaluate
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C  # noqa: E402
from src.models import RNNForecaster  # noqa: E402
from src.train import predict  # noqa: E402
from src.windows import Scaler, build_arrays, gather, make_anchors  # noqa: E402

METHODS = ["lstm", "gru", "hist", "persist"]
LABELS = {"lstm": "LSTM", "gru": "GRU", "hist": "Promedio histórico", "persist": "Persistencia"}


def load_models(device="cpu"):
    models = {}
    for cell in ["lstm", "gru"]:
        p = C.MODEL_DIR / f"{cell}.pt"
        if p.exists():
            m = RNNForecaster(cell)
            m.load_state_dict(torch.load(p, map_location=device))
            models[cell] = m.to(device).eval()
    return models


def model_is_residual() -> bool:
    p = C.MODEL_DIR / "model_meta.json"
    return json.loads(p.read_text()).get("residual", False) if p.exists() else False


def vehicle_history_rate(trips: pd.DataFrame, fallback: float) -> pd.Series:
    """L/h promedio de los trayectos ANTERIORES del mismo vehículo."""
    t = trips.sort_values(["veh", "daynum", "trip"]).copy()
    t["hours"] = t["steps"] * C.STEP_S / 3600.0
    g = t.groupby("veh")
    prev_l = g["liters"].cumsum() - t["liters"]
    prev_h = g["hours"].cumsum() - t["hours"]
    rate = np.where(prev_h > 0, prev_l / prev_h.replace(0, np.nan), fallback)
    return pd.Series(rate, index=pd.MultiIndex.from_frame(t[["veh", "trip"]]))


def persistence(arr, starts, ts, fallback):
    """Media del consumo observado en la historia disponible (máx. LOOKBACK pasos)."""
    hist = arr.fcr[starts[:, None] + np.arange(C.LOOKBACK)[None, :]]
    n_obs = np.minimum(ts, C.LOOKBACK)
    s = hist.sum(1)
    return np.where(n_obs > 0, s / np.maximum(n_obs, 1), fallback)


def metrics(y, p):
    e = p - y
    return {"MAE": float(np.mean(np.abs(e))), "RMSE": float(np.sqrt(np.mean(e ** 2))),
            "bias": float(np.mean(e))}


def trip_metrics(real, pred):
    e = pred - real
    ape = np.abs(e) / np.maximum(real, 1e-6)
    return {"MAE_L": float(np.mean(np.abs(e))), "RMSE_L": float(np.sqrt(np.mean(e ** 2))),
            "MAPE_pct": float(100 * np.median(ape)), "MAPE_mean_pct": float(100 * np.mean(ape)),
            "bias_L": float(np.mean(e)),
            "total_error_pct": float(100 * (pred.sum() - real.sum()) / real.sum())}


def forecast_split(bins, trips, scaler, models, train_rate, device):
    arr = build_arrays(bins, scaler)
    starts, target, tid, ts, blen, pers = make_anchors(arr, C.HORIZON, False, scaler.y_mean)
    keys = np.array(arr.trip_keys)
    df = pd.DataFrame({"veh": keys[tid, 0], "trip": keys[tid, 1], "t": ts, "block_len": blen,
                       "real": target})
    X_dev = torch.as_tensor(arr.X, device=device)
    residual = model_is_residual()
    for cell, m in models.items():
        df[cell] = predict(m, X_dev, starts, scaler, device, base=pers if residual else None)
    hist_rate = vehicle_history_rate(trips, train_rate)
    df["hist"] = hist_rate.reindex(pd.MultiIndex.from_frame(df[["veh", "trip"]])).to_numpy()
    df["persist"] = persistence(arr, starts, ts, df["hist"].to_numpy())
    # Litros por bloque = tasa (L/h) * duración del bloque (h)
    hrs = df["block_len"] * C.STEP_S / 3600.0
    for m in ["real"] + [k for k in METHODS if k in df]:
        df[f"L_{m}"] = df[m] * hrs
    tp = df.groupby(["veh", "trip"])[[c for c in df if c.startswith("L_")]].sum().reset_index()
    tp = trips.merge(tp, on=["veh", "trip"])
    return df, tp, arr, starts


def time_inference(models, arr, starts, n=300):
    out = {}
    cpu = torch.device("cpu")
    torch.set_num_threads(max(1, torch.get_num_threads()))
    xb1 = torch.as_tensor(gather(arr.X, starts[:1]))
    xbB = torch.as_tensor(gather(arr.X, starts[:1024]))
    for cell, m in models.items():
        m = m.to(cpu).eval()
        with torch.no_grad():
            for _ in range(20):
                m(xb1)
            t0 = time.perf_counter()
            for _ in range(n):
                m(xb1)
            lat = (time.perf_counter() - t0) / n * 1000
            t0 = time.perf_counter()
            for _ in range(5):
                m(xbB)
            thr = 5 * len(xbB) / (time.perf_counter() - t0)
        out[cell] = {"latency_ms_cpu_batch1": round(lat, 3), "throughput_per_s_cpu_b1024": round(thr)}
    return out


def log_iteration(results):
    """Agrega una fila a artifacts/iterations.csv (historial de experimentos)."""
    ts = json.loads((C.ART_DIR / "training_summary.json").read_text())
    cfg = ts["config"]
    row = {"timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
           "target": cfg.get("target"), "features": "+".join(cfg["FEATURES"]), "lookback": cfg["LOOKBACK"], "horizon": cfg["HORIZON"],
           "hidden": cfg["HIDDEN"], "layers": cfg["NUM_LAYERS"], "quick": cfg.get("quick")}
    for m, v in results["block"]["test"].items():
        row[f"{m}_block_MAE"] = round(v["MAE"], 4)
        row[f"{m}_block_RMSE"] = round(v["RMSE"], 4)
    for m, v in results["trip"]["test"].items():
        row[f"{m}_trip_MAPE"] = round(v["MAPE_pct"], 3)
    p = C.ART_DIR / "iterations.csv"
    df = pd.concat([pd.read_csv(p), pd.DataFrame([row])]) if p.exists() else pd.DataFrame([row])
    df.to_csv(p, index=False)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scaler = Scaler.from_json(C.MODEL_DIR / "scaler.json")
    models = load_models(device)
    if not models:
        raise SystemExit("No hay modelos. Ejecuta primero: python -m src.train")
    bins = pd.read_parquet(C.PROC_DIR / "bins.parquet")
    trips = pd.read_parquet(C.PROC_DIR / "trips.parquet")
    tr = trips[trips["split"] == "train"]
    train_rate = float(tr["liters"].sum() / (tr["steps"].sum() * C.STEP_S / 3600.0))

    results = {"train_mean_rate_Lh": train_rate, "block": {}, "trip": {}}
    for split in ["val", "test"]:
        b = bins[bins["split"] == split]
        t = trips[trips["split"] == split]
        blocks, tp, arr, starts = forecast_split(b, t, scaler, models, train_rate, device)
        blocks.to_parquet(C.ART_DIR / f"blocks_{split}.parquet", index=False)
        tp.to_parquet(C.ART_DIR / f"trips_{split}.parquet", index=False)
        avail = [m for m in METHODS if m in blocks]
        results["block"][split] = {m: metrics(blocks["real"].to_numpy(), blocks[m].to_numpy()) for m in avail}
        results["trip"][split] = {m: trip_metrics(tp["L_real"].to_numpy(), tp[f"L_{m}"].to_numpy()) for m in avail}
        results[f"n_{split}"] = {"vehicles": int(t["veh"].nunique()), "trips": int(len(t)),
                                 "blocks": int(len(blocks)), "liters": float(tp["L_real"].sum())}
        if split == "test":
            results["inference"] = time_inference(models, arr, starts)
            b[["veh", "trip", "step", "speed", "rpm", "load", "fcr"]].to_parquet(
                C.ART_DIR / "test_bins.parquet", index=False)

    (C.ART_DIR / "metrics.json").write_text(json.dumps(results, indent=2))
    log_iteration(results)
    for split in ["val", "test"]:
        print(f"\n=== {split.upper()}: pronóstico cada {C.HORIZON*C.STEP_S//60} min (L/h) ===")
        print(pd.DataFrame(results["block"][split]).T.round(3).rename(index=LABELS).to_string())
        print(f"--- {split.upper()}: litros por trayecto ---")
        print(pd.DataFrame(results["trip"][split]).T.round(3).rename(index=LABELS).to_string())
    print("\nInferencia:", json.dumps(results["inference"], indent=1))


if __name__ == "__main__":
    main()
