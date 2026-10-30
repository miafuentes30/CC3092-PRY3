"""Modelo de valor: traduce error de predicción a valor de negocio.

Decisión: antes/durante el trayecto se reserva R = P · (1 + m) litros, donde P es el
consumo predicho del trayecto y m el margen de reserva.

    valor = ahorro por acierto − costo de déficit − costo de exceso

  ahorro por acierto : S                          si R ≥ A  (la reserva alcanzó: se evita una
                                                  emergencia de reabastecimiento)
  costo de déficit   : F + φ · p · (A − R)        si R < A  (recargo fijo + prima por litro)
  costo de exceso    : h · p · (R − A)            si R > A  (capital inmovilizado)

A = litros reales, p = precio por litro. El margen m* se elige en VALIDACIÓN y se
reporta en PRUEBA (no se optimiza sobre prueba). Todos los parámetros son supuestos
documentados en config.py y se pueden cambiar en la app.

Uso:
    python -m src.value_model
"""
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C  # noqa: E402


@dataclass
class ValueParams:
    price: float = C.FUEL_PRICE_USD_L
    saving_hit: float = C.SAVING_PER_HIT_USD
    deficit_fixed: float = C.DEFICIT_FIXED_USD
    deficit_premium: float = C.DEFICIT_PREMIUM_FRAC
    excess_frac: float = C.EXCESS_COST_FRAC


def margins():
    lo, hi, st = C.MARGIN_GRID
    return np.round(np.arange(lo, hi + st / 2, st), 4)


def trip_value(A, R, prm: ValueParams):
    hit = R >= A
    D = np.maximum(A - R, 0.0)
    E = np.maximum(R - A, 0.0)
    return (prm.saving_hit * hit
            - (prm.deficit_fixed + prm.deficit_premium * prm.price * D) * (~hit)
            - prm.excess_frac * prm.price * E)


def value_curve(A, P, prm: ValueParams, ms=None):
    """Valor medio por trayecto y tasa de cobertura para cada margen."""
    ms = margins() if ms is None else ms
    R = P[None, :] * (1.0 + ms[:, None])
    v = trip_value(A[None, :], R, prm)
    return pd.DataFrame({"margin": ms, "value_per_trip": v.mean(1),
                         "coverage": (R >= A[None, :]).mean(1)})


def outcome(A, P, m, prm: ValueParams):
    R = P * (1 + m)
    hit = R >= A
    v = trip_value(A, R, prm)
    return {"margin": float(m), "value_per_trip": float(v.mean()), "value_total": float(v.sum()),
            "coverage_pct": float(100 * hit.mean()), "emergencies": int((~hit).sum()),
            "excess_L_per_trip": float(np.maximum(R - A, 0).mean()),
            "deficit_L_per_trip": float(np.maximum(A - R, 0).mean()),
            "reserved_L_per_trip": float(R.mean()), "n_trips": int(len(A))}


def error_value_curve(A, prm: ValueParams, sigmas=None, seed=C.SEED):
    """Curva de valor por nivel de error: predicciones sintéticas P = A·exp(σz),
    margen óptimo para cada σ, valor esperado resultante."""
    sigmas = np.round(np.arange(0.0, 0.81, 0.02), 3) if sigmas is None else sigmas
    z = np.random.default_rng(seed).standard_normal(len(A))
    rows = []
    for s in sigmas:
        P = A * np.exp(s * z)
        vc = value_curve(A, P, prm)
        best = vc.loc[vc["value_per_trip"].idxmax()]
        rows.append({"sigma": float(s), "best_margin": float(best["margin"]),
                     "value_per_trip": float(best["value_per_trip"]),
                     "coverage": float(best["coverage"])})
    return pd.DataFrame(rows)


def log_error_sigma(A, P):
    r = np.log(np.maximum(P, 1e-6) / np.maximum(A, 1e-6))
    return float(np.std(r))


def analyze(trips_val, trips_test, methods, prm: ValueParams):
    out = {"params": asdict(prm), "methods": {}}
    Av, At = trips_val["L_real"].to_numpy(), trips_test["L_real"].to_numpy()
    curves = {}
    for m in methods:
        Pv, Pt = trips_val[f"L_{m}"].to_numpy(), trips_test[f"L_{m}"].to_numpy()
        cv = value_curve(Av, Pv, prm)
        m_star = float(cv.loc[cv["value_per_trip"].idxmax(), "margin"])
        curves[m] = value_curve(At, Pt, prm)
        out["methods"][m] = {"margin_selected_on_val": m_star,
                             "test": outcome(At, Pt, m_star, prm),
                             "test_log_error_sigma": log_error_sigma(At, Pt)}
    out["perfect_info_value_per_trip"] = float(prm.saving_hit)
    return out, curves


def main():
    tv = pd.read_parquet(C.ART_DIR / "trips_val.parquet")
    tt = pd.read_parquet(C.ART_DIR / "trips_test.parquet")
    methods = [m for m in ["lstm", "gru", "hist", "persist"] if f"L_{m}" in tt]
    prm = ValueParams()
    res, _ = analyze(tv, tt, methods, prm)
    base = res["methods"]["hist"]["test"]
    for m in methods:
        t = res["methods"][m]["test"]
        t["gain_vs_hist_per_trip"] = t["value_per_trip"] - base["value_per_trip"]
        t["gain_vs_hist_per_1000_trips"] = 1000 * t["gain_vs_hist_per_trip"]
    ec = error_value_curve(tt["L_real"].to_numpy(), prm)
    ec.to_csv(C.ART_DIR / "error_value_curve.csv", index=False)
    (C.ART_DIR / "value_results.json").write_text(json.dumps(res, indent=2))
    rows = {m: {"σ log-error": res["methods"][m]["test_log_error_sigma"],
                "margen (val)": res["methods"][m]["margin_selected_on_val"],
                **{k: res["methods"][m]["test"][k] for k in
                   ["value_per_trip", "coverage_pct", "emergencies", "excess_L_per_trip",
                    "gain_vs_hist_per_1000_trips"]}} for m in methods}
    print(pd.DataFrame(rows).T.round(3).to_string())


if __name__ == "__main__":
    main()
