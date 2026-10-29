"""Preprocesamiento del VED para FuelWise AI.

Pasos:
 1. Filtra los vehículos de gasolina (Vehicle Type == ICE).
 2. Estima la tasa de consumo de combustible (FCR, L/h) por registro con el
    Algoritmo 1 del paper de VED (Fuel Rate -> MAF -> carga/RPM/cilindrada).
 3. Agrega cada trayecto a pasos regulares de STEP_S segundos.
 4. Verifica calidad: interpola huecos cortos y descarta trayectos con pocas
    mediciones válidas o demasiado cortos.
 5. Divide por VEHÍCULO (trayectos completos) en train/val/test ANTES de construir
    ventanas, y guarda todo en data/processed/.

Uso:
    python -m src.preprocess
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C  # noqa: E402

RAW_COLS = {
    "DayNum": "daynum",
    "VehId": "veh",
    "Trip": "trip",
    "Timestamp(ms)": "ts_ms",
    "Vehicle Speed[km/h]": "speed",
    "MAF[g/sec]": "maf",
    "Engine RPM[RPM]": "rpm",
    "Absolute Load[%]": "load",
    "OAT[DegC]": "oat",
    "Fuel Rate[L/hr]": "fuel_rate",
    "Short Term Fuel Trim Bank 1[%]": "stft1",
    "Short Term Fuel Trim Bank 2[%]": "stft2",
    "Long Term Fuel Trim Bank 1[%]": "ltft1",
    "Long Term Fuel Trim Bank 2[%]": "ltft2",
}
SRC_NAMES = {0: "Fuel Rate OBD", 1: "MAF + fuel trims", 2: "Carga·RPM·cilindrada", -1: "Sin dato"}


# --------------------------------------------------------------------------
# Datos estáticos
# --------------------------------------------------------------------------
def parse_displacement(s) -> float:
    m = re.search(r"(\d+(?:\.\d+)?)\s*L", str(s))
    return float(m.group(1)) if m else np.nan


def load_static() -> pd.DataFrame:
    st = pd.read_excel(C.RAW_DIR / "VED_Static_Data_ICE&HEV.xlsx")
    st = st.rename(columns={"VehId": "veh", "Vehicle Type": "vtype", "Vehicle Class": "vclass",
                            "Engine Configuration & Displacement": "engine",
                            "Generalized_Weight": "weight_lb"})
    st["disp_l"] = st["engine"].map(parse_displacement)
    st["weight_lb"] = pd.to_numeric(st["weight_lb"], errors="coerce")
    return st[["veh", "vtype", "vclass", "engine", "disp_l", "weight_lb", "Transmission"]]


# --------------------------------------------------------------------------
# Algoritmo 1 (VED): estimación de la tasa de consumo de combustible
# --------------------------------------------------------------------------
def estimate_fcr(df: pd.DataFrame, disp_l) -> pd.DataFrame:
    """Agrega columnas `fcr` (L/h) y `fcr_src` a un DataFrame con columnas crudas
    renombradas. `disp_l` puede ser escalar o Serie alineada (cilindrada en litros).

    correction = (1 + STFT/100 + LTFT/100) / AFR          (promedio de bancos si hay 2)
    combustible[g/s] = MAF * correction
    si no hay MAF:  MAF = Load/100 * rho_aire * cilindrada * RPM / 120
    FCR[L/h] = combustible[g/s] * 3600 / densidad_gasolina
    """
    for c in ["stft1", "stft2", "ltft1", "ltft2", "maf", "load", "fuel_rate"]:
        if c not in df:
            df[c] = np.nan
    stft = df[["stft1", "stft2"]].mean(axis=1).fillna(0.0)
    ltft = df[["ltft1", "ltft2"]].mean(axis=1).fillna(0.0)
    correction = (1.0 + stft / 100.0 + ltft / 100.0) / C.AFR_E10

    maf_est = df["load"] / 100.0 * C.AIR_DENSITY_G_L * disp_l * df["rpm"] / 120.0
    g_s_from_maf = df["maf"] * correction
    g_s_from_load = maf_est * correction
    to_lh = 3600.0 / C.FUEL_DENSITY_G_L

    fcr = np.where(df["fuel_rate"].notna(), df["fuel_rate"],
          np.where(df["maf"].notna(), g_s_from_maf * to_lh,
          np.where(np.isfinite(g_s_from_load), g_s_from_load * to_lh, np.nan)))
    src = np.where(df["fuel_rate"].notna(), 0,
          np.where(df["maf"].notna(), 1,
          np.where(np.isfinite(g_s_from_load), 2, -1)))
    df["fcr"] = np.clip(fcr, 0.0, C.FCR_MAX_LH)
    df["fcr_src"] = src
    return df


def clean_signals(df: pd.DataFrame) -> pd.DataFrame:
    df["speed"] = df["speed"].clip(0, 200)
    df["rpm"] = df["rpm"].clip(0, 8000)
    df["load"] = df["load"].clip(0, 200)
    return df


# --------------------------------------------------------------------------
# Agregación a pasos regulares
# --------------------------------------------------------------------------
AGG_VARS = ["speed", "rpm", "load", "fcr", "oat"]


def bin_records(df: pd.DataFrame) -> pd.DataFrame:
    """Suma y conteo por (veh, trip, paso) para poder recombinar archivos."""
    df["step"] = (df["ts_ms"] // (C.STEP_S * 1000)).astype(np.int64)
    g = df.groupby(["veh", "trip", "step"], sort=False)
    out = g[AGG_VARS].sum(min_count=1)
    cnt = g[AGG_VARS].count().add_suffix("_n")
    out = out.join(cnt)
    out["daynum"] = g["daynum"].min()
    return out.reset_index()


def regularize_trip(tr: pd.DataFrame) -> pd.DataFrame:
    """Reindexa un trayecto a pasos continuos 0..T-1 e interpola huecos cortos."""
    tr = tr.set_index("step").sort_index()
    full = np.arange(tr.index.min(), tr.index.max() + 1)
    tr = tr.reindex(full)
    for v in ["speed", "rpm", "load", "fcr"]:
        tr[v] = tr[v].interpolate(limit=C.MAX_GAP_STEPS, limit_area="inside")
    tr["valid"] = tr[["speed", "rpm", "load", "fcr"]].notna().all(axis=1)
    tr.index = tr.index - tr.index.min()
    return tr.rename_axis("step").reset_index()


def process_records_to_bins(rec: pd.DataFrame, disp_l) -> pd.DataFrame:
    """Para un único trayecto subido por el usuario (lo usa la app)."""
    rec = clean_signals(estimate_fcr(rec, disp_l))
    if "veh" not in rec:
        rec["veh"] = 0
    if "trip" not in rec:
        rec["trip"] = 0
    if "daynum" not in rec:
        rec["daynum"] = 0.0
    b = bin_records(rec)
    for v in AGG_VARS:
        b[v] = b[v] / b[v + "_n"].replace(0, np.nan)
    tr = regularize_trip(b[["step"] + AGG_VARS])
    for v in ["speed", "rpm", "load", "fcr"]:
        tr[v] = tr[v].ffill().bfill()
    return tr


# --------------------------------------------------------------------------
# Pipeline principal
# --------------------------------------------------------------------------
def main() -> None:
    C.PROC_DIR.mkdir(parents=True, exist_ok=True)
    static = load_static()
    ice = static[static["vtype"] == "ICE"]
    # El único ICE diésel (VehId 119, '8-DSL 6.7L') se excluye: el AFR de E10 no aplica.
    ice = ice[~ice["engine"].astype(str).str.contains("DSL", case=False)].set_index("veh")
    print(f"Vehículos de gasolina (ICE, sin diésel) en datos estáticos: {len(ice)}")

    files = sorted(C.CSV_DIR.glob("*.csv"))
    if not files:
        raise SystemExit("No hay CSV. Ejecuta primero: python -m src.download_data")

    parts, n_rows, src_counts, raw_cov = [], 0, {}, []
    for i, f in enumerate(files, 1):
        df = pd.read_csv(f, usecols=list(RAW_COLS)).rename(columns=RAW_COLS)
        df = df[df["veh"].isin(ice.index)].copy()
        if df.empty:
            continue
        n_rows += len(df)
        raw_cov.append(df[["speed", "rpm", "load", "maf", "fuel_rate", "oat"]].notna().sum())
        disp = df["veh"].map(ice["disp_l"])
        df = clean_signals(estimate_fcr(df, disp))
        for k, v in df["fcr_src"].value_counts().items():
            src_counts[int(k)] = src_counts.get(int(k), 0) + int(v)
        parts.append(bin_records(df))
        print(f"  [{i}/{len(files)}] {f.name}: {len(df):,} registros ICE")

    bins = pd.concat(parts, ignore_index=True)
    # recombinar pasos que aparezcan en más de un archivo
    sums = bins.groupby(["veh", "trip", "step"])[AGG_VARS + [v + "_n" for v in AGG_VARS]].sum(min_count=1)
    sums["daynum"] = bins.groupby(["veh", "trip", "step"])["daynum"].min()
    for v in AGG_VARS:
        sums[v] = sums[v] / sums[v + "_n"].replace(0, np.nan)
    bins = sums[AGG_VARS + ["daynum"]].reset_index()
    print(f"Pasos de {C.STEP_S}s antes de filtrar: {len(bins):,}")

    # Regularizar y filtrar trayectos por calidad
    kept, stats = [], {"trips_total": 0, "trips_short": 0, "trips_low_quality": 0}
    for (veh, trip), tr in bins.groupby(["veh", "trip"], sort=False):
        stats["trips_total"] += 1
        tr = regularize_trip(tr.drop(columns=["veh", "trip"]))
        if len(tr) < C.MIN_TRIP_STEPS:
            stats["trips_short"] += 1
            continue
        if tr["valid"].mean() < C.MIN_VALID_FRAC:
            stats["trips_low_quality"] += 1
            continue
        for v in ["speed", "rpm", "load", "fcr"]:
            tr[v] = tr[v].ffill().bfill()
        tr["daynum"] = tr["daynum"].min()
        tr.insert(0, "trip", trip)
        tr.insert(0, "veh", veh)
        kept.append(tr)
    data = pd.concat(kept, ignore_index=True)
    data = data[["veh", "trip", "step", "daynum", "speed", "rpm", "load", "fcr", "oat"]]
    data[["speed", "rpm", "load", "fcr", "oat"]] = data[["speed", "rpm", "load", "fcr", "oat"]].astype("float32")

    # Resumen por trayecto
    h = C.STEP_S / 3600.0
    trips = data.groupby(["veh", "trip"]).agg(
        daynum=("daynum", "first"), steps=("step", "size"),
        liters=("fcr", lambda x: float(x.sum() * h)),
        distance_km=("speed", lambda x: float(x.sum() * h)),
        mean_fcr=("fcr", "mean"), mean_speed=("speed", "mean")).reset_index()
    trips["duration_min"] = trips["steps"] * C.STEP_S / 60.0

    # División por vehículo (trayectos completos, sin vehículos compartidos)
    rng = np.random.default_rng(C.SEED)
    vehs = np.array(sorted(trips["veh"].unique()))
    rng.shuffle(vehs)
    n_tr = int(round(len(vehs) * C.SPLIT_FRAC["train"]))
    n_va = int(round(len(vehs) * C.SPLIT_FRAC["val"]))
    split = {int(v): "train" for v in vehs[:n_tr]}
    split.update({int(v): "val" for v in vehs[n_tr:n_tr + n_va]})
    split.update({int(v): "test" for v in vehs[n_tr + n_va:]})
    trips["split"] = trips["veh"].map(split)
    data["split"] = data["veh"].map(split)
    trips = trips.merge(static[["veh", "vclass", "engine", "disp_l", "weight_lb"]], on="veh", how="left")

    data.to_parquet(C.PROC_DIR / "bins.parquet", index=False)
    trips.to_parquet(C.PROC_DIR / "trips.parquet", index=False)

    cov = pd.concat(raw_cov, axis=1).sum(axis=1) / n_rows
    report = {
        "ice_vehicles_static": int(len(ice)),
        "ice_raw_records": int(n_rows),
        "raw_coverage_frac": {k: round(float(v), 4) for k, v in cov.items()},
        "fcr_source_records": {SRC_NAMES[k]: v for k, v in sorted(src_counts.items())},
        **stats,
        "trips_kept": int(len(trips)),
        "vehicles_kept": int(trips["veh"].nunique()),
        "steps_kept": int(len(data)),
        "hours_kept": round(float(len(data) * h), 1),
        "liters_kept": round(float(trips["liters"].sum()), 1),
        "split_vehicles": {s: int(sum(1 for v in split.values() if v == s)) for s in ["train", "val", "test"]},
        "split_trips": trips["split"].value_counts().to_dict(),
        "step_s": C.STEP_S,
    }
    (C.PROC_DIR / "quality_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
