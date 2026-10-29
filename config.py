"""Configuración central de FuelWise AI.

Todas las constantes del pipeline (rutas, ventanas, hiperparámetros y supuestos
económicos) viven aquí para que los resultados sean reproducibles.
"""
from pathlib import Path

# --------------------------------------------------------------------------
# Rutas
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"              # .7z y .xlsx descargados
CSV_DIR = DATA_DIR / "raw" / "csv"      # CSV semanales descomprimidos
PROC_DIR = DATA_DIR / "processed"       # parquet procesados
ART_DIR = ROOT / "artifacts"            # modelos, métricas y predicciones (lo que usa la app)
MODEL_DIR = ART_DIR / "models"

VED_BASE_URL = "https://raw.githubusercontent.com/gsoh/VED/master/Data/"
VED_FILES = [
    "VED_DynamicData_Part1.7z",
    "VED_DynamicData_Part2.7z",
    "VED_Static_Data_ICE&HEV.xlsx",
]

SEED = 42

# --------------------------------------------------------------------------
# Estimación del consumo (Oh et al., VED, Sección II.C, Algoritmo 1)
# --------------------------------------------------------------------------
AFR_E10 = 14.08          # relación aire/combustible estequiométrica para E10 (paper VED)
FUEL_DENSITY_G_L = 745.0  # densidad de gasolina [g/L] (supuesto documentado, ~0.74-0.75 kg/L)
AIR_DENSITY_G_L = 1.225   # densidad del aire [g/L] a nivel del mar, 15 °C (para el respaldo carga/RPM)
FCR_MAX_LH = 100.0        # límite físico para recortar valores anómalos [L/h]

# --------------------------------------------------------------------------
# Series temporales
# --------------------------------------------------------------------------
STEP_S = 10            # resolución de las series: promedio cada 10 s
LOOKBACK = 18          # historia que ve el modelo: 18 pasos = 3 min
HORIZON = 12           # horizonte del pronóstico: 12 pasos = 2 min (se actualiza cada 2 min)
TRAIN_STRIDE = 3       # separación entre ventanas de entrenamiento (30 s)
MAX_GAP_STEPS = 3      # huecos <= 30 s se interpolan dentro del trayecto
MIN_TRIP_STEPS = 18    # trayectos de al menos 3 min
MIN_VALID_FRAC = 0.90  # % mínimo de pasos con todas las variables válidas por trayecto

# Variables de entrada de la propuesta: velocidad, RPM, carga y consumo histórico.
# Iteración 3: el consumo histórico se da también como promedio acumulado del trayecto
# (fcr_tripmean = consumo medio desde que inició el trayecto hasta ese paso).
TRIP_MEAN_FEATURE = False  # iteración 3 probada; no mejoró en validación
FEATURES = ["speed", "rpm", "load", "fcr"] + (["fcr_tripmean"] if TRIP_MEAN_FEATURE else [])
N_INPUTS = len(FEATURES) + 1  # + is_pad

# División por vehículo (los vehículos de prueba no se ven en entrenamiento)
SPLIT_FRAC = {"train": 0.70, "val": 0.15, "test": 0.15}

# --------------------------------------------------------------------------
# Entrenamiento
# --------------------------------------------------------------------------
HIDDEN = 64
NUM_LAYERS = 2
DROPOUT = 0.2
LR = 1e-3
BATCH_SIZE = 1024
MAX_EPOCHS = 30
PATIENCE = 5
# Iteración 2: el modelo predice la CORRECCIÓN sobre la persistencia (consumo medio de
# los últimos 3 min). Iteración 1 (RESIDUAL=False) predice el consumo directamente.
RESIDUAL = True

# --------------------------------------------------------------------------
# Modelo de valor (supuestos por defecto; todos ajustables en la app)
# valor = ahorro por acierto - costo de déficit - costo de exceso
# --------------------------------------------------------------------------
GAL_TO_L = 3.78541
# EIA, Gasoline and Diesel Fuel Update, precio minorista regular Midwest (PADD 2),
# semana del 26-oct-2026: 4.269 USD/gal  -> USD/L
FUEL_PRICE_USD_L = round(4.269 / GAL_TO_L, 4)
# Engineer Fix (dic-2025): cargo base de asistencia en carretera por falta de
# combustible para no miembros: 50-150 USD -> punto medio 100 USD.
# Se usa como "ahorro por acierto": emergencia evitada cuando la reserva alcanza.
SAVING_PER_HIT_USD = 100.0
# Engineer Fix: recargos nocturnos/fin de semana 20-50 USD -> punto medio 35 USD.
# Se usa como costo fijo adicional cuando la reserva no alcanza.
DEFICIT_FIXED_USD = 35.0
# Prima por litro faltante comprado en emergencia (fracción del precio).
# Engineer Fix indica que el combustible entregado se cobra a precio de mercado -> 0.
DEFICIT_PREMIUM_FRAC = 0.0
# Costo de exceso: fracción del precio por litro reservado y no usado (capital
# inmovilizado / manejo). SUPUESTO DEL EQUIPO, sin fuente pública directa.
EXCESS_COST_FRAC = 0.10
MARGIN_GRID = (-0.5, 3.0, 0.01)  # márgenes de reserva evaluados: -50 % a +300 %
