"""Entrena los modelos LSTM y GRU.

- El escalado se ajusta SOLO con los trayectos de entrenamiento.
- Early stopping con el MAE (L/h) del conjunto de validación (vehículos distintos).
- Usa GPU automáticamente si está disponible (p. ej. Google Colab).

Uso:
    python -m src.train                 # entrena LSTM y GRU
    python -m src.train --epochs 5      # menos épocas
    python -m src.train --quick         # prueba rápida con 25 % de los trayectos
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C  # noqa: E402
from src.models import RNNForecaster, count_params, flops_per_forecast  # noqa: E402
from src.windows import Scaler, build_arrays, make_anchors  # noqa: E402


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_split_bins(quick: bool = False):
    bins = pd.read_parquet(C.PROC_DIR / "bins.parquet")
    if quick:
        trips = bins[["veh", "trip"]].drop_duplicates().sample(frac=0.25, random_state=C.SEED)
        bins = bins.merge(trips, on=["veh", "trip"])
    return {s: bins[bins["split"] == s] for s in ["train", "val", "test"]}


@torch.no_grad()
def predict(model, X_dev, starts, scaler, device, base=None, batch=8192):
    """Pronóstico en L/h. Si `base` (persistencia) se pasa, el modelo es residual:
    pred = persistencia + salida·σ_y ; si no, pred = media_y + salida·σ_y."""
    model.eval()
    out = []
    ar = torch.arange(C.LOOKBACK, device=device)
    for i in range(0, len(starts), batch):
        s = torch.as_tensor(starts[i:i + batch], device=device)
        xb = X_dev[s[:, None] + ar[None, :]]
        out.append(model(xb).float().cpu().numpy())
    y = np.concatenate(out) * scaler.y_std + (scaler.y_mean if base is None else base)
    return np.clip(y, 0.0, None)


def train_one(cell, arrays, anchors, scaler, device, epochs, log, residual=C.RESIDUAL):
    set_seed(C.SEED)
    model = RNNForecaster(cell).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=C.LR)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=1)
    loss_fn = torch.nn.MSELoss()

    Xtr = torch.as_tensor(arrays["train"].X, device=device)
    Xva = torch.as_tensor(arrays["val"].X, device=device)
    tr_s, tr_y, tr_p = anchors["train"][0], anchors["train"][1], anchors["train"][5]
    va_s, va_y, va_p = anchors["val"][0], anchors["val"][1], anchors["val"][5]
    tr_base = tr_p if residual else scaler.y_mean
    va_base = va_p if residual else None
    ytr = torch.as_tensor((tr_y - tr_base) / scaler.y_std, device=device)
    ar = torch.arange(C.LOOKBACK, device=device)

    best, best_state, bad, hist = np.inf, None, 0, []
    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        te = time.time()
        perm = torch.randperm(len(tr_s), device=device)
        s_all = torch.as_tensor(tr_s, device=device)
        tot, n = 0.0, 0
        for i in range(0, len(perm), C.BATCH_SIZE):
            idx = perm[i:i + C.BATCH_SIZE]
            xb = Xtr[s_all[idx][:, None] + ar[None, :]]
            loss = loss_fn(model(xb), ytr[idx])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * len(idx); n += len(idx)
        pv = predict(model, Xva, va_s, scaler, device, base=va_base)
        mae = float(np.mean(np.abs(pv - va_y)))
        rmse = float(np.sqrt(np.mean((pv - va_y) ** 2)))
        sched.step(mae)
        hist.append({"epoch": ep, "train_mse_scaled": tot / n, "val_mae": mae, "val_rmse": rmse,
                     "epoch_s": time.time() - te, "lr": opt.param_groups[0]["lr"]})
        log(f"  [{cell.upper()}] época {ep:02d}  loss={tot/n:.4f}  val MAE={mae:.3f} L/h  "
            f"RMSE={rmse:.3f}  ({time.time()-te:.1f}s)")
        if mae < best - 1e-4:
            best, bad = mae, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= C.PATIENCE:
                log(f"  early stopping en época {ep}")
                break
    train_s = time.time() - t0
    model.load_state_dict(best_state)
    info = {"cell": cell, "residual": residual, "params": count_params(model), "flops_per_forecast": flops_per_forecast(cell),
            "train_time_s": round(train_s, 1), "epochs_run": len(hist),
            "best_val_mae": best, "device": str(device), "history": hist}
    return model, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["lstm", "gru"])
    ap.add_argument("--epochs", type=int, default=C.MAX_EPOCHS)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--target", choices=["residual", "direct"], default="residual" if C.RESIDUAL else "direct",
                    help="residual: el modelo aprende la corrección sobre la persistencia")
    args = ap.parse_args()

    C.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")
    set_seed(C.SEED)

    splits = load_split_bins(args.quick)
    scaler = Scaler.fit(splits["train"])               # solo entrenamiento
    scaler.to_json(C.MODEL_DIR / "scaler.json")
    arrays = {s: build_arrays(splits[s], scaler) for s in ["train", "val"]}
    fb = scaler.y_mean
    anchors = {"train": make_anchors(arrays["train"], C.TRAIN_STRIDE, True, fb),
               "val": make_anchors(arrays["val"], C.HORIZON, False, fb)}
    residual = args.target == "residual"
    print(f"Ventanas de entrenamiento: {len(anchors['train'][0]):,} | "
          f"bloques de validación: {len(anchors['val'][0]):,}")

    summary = {}
    for cell in args.models:
        model, info = train_one(cell, arrays, anchors, scaler, device, args.epochs, print, residual)
        torch.save(model.state_dict(), C.MODEL_DIR / f"{cell}.pt")
        summary[cell] = info
    cfg = {k: getattr(C, k) for k in ["STEP_S", "LOOKBACK", "HORIZON", "TRAIN_STRIDE", "HIDDEN",
                                      "NUM_LAYERS", "DROPOUT", "LR", "BATCH_SIZE", "MAX_EPOCHS",
                                      "PATIENCE", "FEATURES", "SEED"]}
    cfg["quick"] = args.quick
    cfg["target"] = args.target
    (C.MODEL_DIR / "model_meta.json").write_text(json.dumps({"residual": residual, "features": C.FEATURES}))
    (C.ART_DIR / "training_summary.json").write_text(json.dumps({"config": cfg, "models": summary}, indent=2))
    print("Modelos guardados en", C.MODEL_DIR)


if __name__ == "__main__":
    main()
