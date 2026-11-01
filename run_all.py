"""Ejecuta el pipeline completo de FuelWise AI.

    python run_all.py                 # descarga, preprocesa, entrena, evalúa y calcula valor
    python run_all.py --skip-download # si ya tienes los CSV en data/raw/csv
    python run_all.py --quick         # prueba rápida (25 % de los trayectos, 3 épocas)
"""
import argparse
import subprocess
import sys

import config as C


def run(mod, *args):
    cmd = [sys.executable, "-m", mod, *args]
    print("\n>>>", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=C.ROOT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--skip-preprocess", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--target", choices=["residual", "direct"], default="residual")
    a = ap.parse_args()
    if not a.skip_download:
        run("src.download_data")
    if not a.skip_preprocess or not (C.PROC_DIR / "bins.parquet").exists():
        run("src.preprocess")
    run("src.train", "--target", a.target, *(["--quick", "--epochs", "3"] if a.quick else []))
    run("src.evaluate")
    run("src.value_model")
    # copia del reporte de calidad para que la app funcione sin data/processed
    (C.ART_DIR / "quality_report.json").write_text((C.PROC_DIR / "quality_report.json").read_text())
    print("\nListo. Inicia la app con:  streamlit run app/app.py")


if __name__ == "__main__":
    main()
