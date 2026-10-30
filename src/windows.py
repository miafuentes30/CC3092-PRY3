"""Construcción de ventanas temporales sin fuga de información.

Para un instante de pronóstico t (en pasos) el modelo ve SOLO los pasos [t-LOOKBACK, t)
del mismo trayecto (si t < LOOKBACK la historia se rellena con ceros y una
máscara `is_pad`=1). El objetivo es el consumo medio (L/h) de los próximos HORIZON
pasos [t, t+HORIZON). Nunca se usan velocidad, RPM ni carga futuras.

Implementación vectorizada: todos los trayectos de un split se concatenan en un
arreglo, cada uno precedido por LOOKBACK filas de relleno; una ventana es un
índice de inicio dentro de ese arreglo.
"""
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C  # noqa: E402


def add_derived(bins: pd.DataFrame) -> pd.DataFrame:
    """Variables derivadas que solo usan información pasada del mismo trayecto."""
    if "fcr_tripmean" in C.FEATURES and "fcr_tripmean" not in bins:
        bins = bins.copy()
        g = bins.groupby(["veh", "trip"], sort=False)["fcr"]
        bins["fcr_tripmean"] = g.cumsum() / (g.cumcount() + 1)
    return bins


@dataclass
class Scaler:
    mean: np.ndarray
    std: np.ndarray
    y_mean: float
    y_std: float

    @classmethod
    def fit(cls, train_bins: pd.DataFrame) -> "Scaler":
        train_bins = add_derived(train_bins)
        x = train_bins[C.FEATURES].to_numpy(np.float64)
        return cls(x.mean(0), x.std(0) + 1e-6,
                   float(train_bins["fcr"].mean()), float(train_bins["fcr"].std()))

    def to_json(self, path: Path) -> None:
        path.write_text(json.dumps({"features": C.FEATURES, "mean": self.mean.tolist(),
                                    "std": self.std.tolist(), "y_mean": self.y_mean,
                                    "y_std": self.y_std}, indent=2))

    @classmethod
    def from_json(cls, path: Path) -> "Scaler":
        d = json.loads(Path(path).read_text())
        return cls(np.array(d["mean"]), np.array(d["std"]), d["y_mean"], d["y_std"])


@dataclass
class SplitArrays:
    X: np.ndarray          # (N_total, N_INPUTS) float32, incluye filas de relleno
    fcr: np.ndarray        # (N_total,) consumo real L/h (0 en relleno)
    trip_keys: list        # [(veh, trip)]
    offsets: np.ndarray    # índice donde empieza (relleno incluido) cada trayecto
    lengths: np.ndarray    # longitud real de cada trayecto


def build_arrays(bins: pd.DataFrame, scaler: Scaler) -> SplitArrays:
    L = C.LOOKBACK
    bins = add_derived(bins)
    xs, ys, keys, offs, lens = [], [], [], [], []
    pos = 0
    for (veh, trip), tr in bins.groupby(["veh", "trip"], sort=False):
        f = (tr[C.FEATURES].to_numpy(np.float64) - scaler.mean) / scaler.std
        n = len(f)
        block = np.zeros((L + n, C.N_INPUTS), np.float32)
        block[:L, -1] = 1.0                  # is_pad
        block[L:, :-1] = f
        y = np.zeros(L + n, np.float32)
        y[L:] = tr["fcr"].to_numpy(np.float32)
        xs.append(block); ys.append(y)
        keys.append((int(veh), int(trip))); offs.append(pos); lens.append(n)
        pos += L + n
    return SplitArrays(np.concatenate(xs), np.concatenate(ys), keys,
                       np.array(offs, np.int64), np.array(lens, np.int64))


def make_anchors(arr: SplitArrays, stride: int, full_horizon_only: bool, fallback: float = 0.0):
    """Devuelve (start_idx, target_mean_fcr, trip_idx, t, block_len, persistence).

    persistence: consumo medio observado en la historia disponible (máx. LOOKBACK
    pasos); si t = 0 no hay historia y se usa `fallback` (media de entrenamiento).

    start_idx: índice en arr.X donde empieza la historia de LOOKBACK pasos.
    Con stride=HORIZON y full_horizon_only=False se obtienen los bloques
    consecutivos que usa la inferencia (el último bloque puede ser parcial).
    """
    L, H = C.LOOKBACK, C.HORIZON
    cs = np.concatenate([[0.0], np.cumsum(arr.fcr, dtype=np.float64)])
    starts, targets, tids, ts, blens, pers = [], [], [], [], [], []
    for i, (off, n) in enumerate(zip(arr.offsets, arr.lengths)):
        last = n - H if full_horizon_only else n - 1
        if last < 0:
            continue
        t = np.arange(0, last + 1, stride)
        b = np.minimum(H, n - t)
        a = off + L + t                     # primer paso futuro (en arr)
        tgt = (cs[a + b] - cs[a]) / b
        n_obs = np.minimum(t, L)
        p = np.where(n_obs > 0, (cs[a] - cs[a - n_obs]) / np.maximum(n_obs, 1), fallback)
        starts.append(off + t)              # historia: [off+t, off+t+L) == pasos [t-L, t)
        targets.append(tgt); tids.append(np.full(len(t), i)); ts.append(t); blens.append(b)
        pers.append(p)
    return (np.concatenate(starts), np.concatenate(targets).astype(np.float32),
            np.concatenate(tids), np.concatenate(ts), np.concatenate(blens),
            np.concatenate(pers).astype(np.float32))


def gather(X, starts):
    idx = starts[:, None] + np.arange(C.LOOKBACK)[None, :]
    return X[idx]
