"""Descarga y descomprime el Vehicle Energy Dataset (VED) desde el repositorio oficial.

Uso:
    python -m src.download_data
"""
import sys
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C  # noqa: E402


def download(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  ya existe: {dest.name}")
        return
    print(f"  descargando {dest.name} ...")
    tmp = dest.with_suffix(dest.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.rename(dest)


def main() -> None:
    C.RAW_DIR.mkdir(parents=True, exist_ok=True)
    C.CSV_DIR.mkdir(parents=True, exist_ok=True)
    for name in C.VED_FILES:
        download(C.VED_BASE_URL + urllib.parse.quote(name), C.RAW_DIR / name)

    if len(list(C.CSV_DIR.glob("*.csv"))) >= 50:
        print("  CSV ya descomprimidos")
        return
    import py7zr
    for name in C.VED_FILES:
        if name.endswith(".7z"):
            print(f"  descomprimiendo {name} ...")
            with py7zr.SevenZipFile(C.RAW_DIR / name) as z:
                z.extractall(C.CSV_DIR)
    # algunos archivos traen subcarpetas: aplanar
    for f in C.CSV_DIR.rglob("*.csv"):
        if f.parent != C.CSV_DIR:
            f.rename(C.CSV_DIR / f.name)
    print(f"Listo: {len(list(C.CSV_DIR.glob('*.csv')))} archivos CSV semanales en {C.CSV_DIR}")


if __name__ == "__main__":
    main()
