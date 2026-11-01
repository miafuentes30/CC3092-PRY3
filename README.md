# FuelWise AI — Predicción del consumo de combustible con LSTM y GRU

Aplicación web que pronostica el consumo de combustible de vehículos de gasolina a partir de series
temporales de conducción (velocidad, RPM, carga del motor y consumo histórico), compara redes **LSTM** y
**GRU** contra métodos simples de referencia y traduce el error de predicción a **valor de negocio** al
decidir cuánto combustible reservar por trayecto.

Datos: [Vehicle Energy Dataset (VED)](https://github.com/gsoh/VED), licencia Apache 2.0.
Equipo: Mia Fuentes (23775) · Roberto Barreda (23354) — Deep Learning y Sistemas Inteligentes, UVG.

---

## 1. Inicio rápido (la app con los modelos ya entrenados)

El zip incluye `artifacts/` con los modelos entrenados y las predicciones de prueba, así que la app
funciona **sin descargar el dataset y sin GPU**.

```bash
cd fuelwise
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app/app.py
```

Se abre en `http://localhost:8501`.

## 2. Reproducir todo el pipeline

```bash
python run_all.py            # descarga VED (~176 MB), preprocesa, entrena, evalúa y calcula el valor
python run_all.py --quick    # prueba rápida: 25 % de los trayectos y 3 épocas
```

O paso a paso:

| Paso | Comando | Salida |
|---|---|---|
| Descarga | `python -m src.download_data` | `data/raw/` (54 CSV semanales, ~3 GB) |
| Preprocesamiento | `python -m src.preprocess` | `data/processed/bins.parquet`, `trips.parquet`, `quality_report.json` |
| Entrenamiento | `python -m src.train [--target residual\|direct]` | `artifacts/models/{lstm,gru}.pt`, `scaler.json`, `training_summary.json` |
| Evaluación | `python -m src.evaluate` | `artifacts/metrics.json`, `blocks_*.parquet`, `trips_*.parquet`, `iterations.csv` |
| Modelo de valor | `python -m src.value_model` | `artifacts/value_results.json`, `error_value_curve.csv` |

**¿Se necesita GPU?** No es obligatoria. Los modelos son pequeños (41–54 mil parámetros).
En CPU (2 núcleos) el entrenamiento de ambos modelos tomó ~9 min; el preprocesamiento ~6–8 min.
Si quieren entrenar más rápido o probar más iteraciones, usen **Google Colab con GPU T4** y el notebook
`notebooks/FuelWise_Colab.ipynb` (instrucciones abajo). El código usa CUDA automáticamente si existe.
La app siempre corre en CPU (latencia < 1 ms por pronóstico).

### Google Colab
1. Abran `notebooks/FuelWise_Colab.ipynb` en Colab (Archivo → Subir notebook).
2. `Entorno de ejecución → Cambiar tipo de entorno de ejecución → T4 GPU`.
3. Ejecuten las celdas: suben `fuelwise.zip`, corre `run_all.py` y se descarga `artifacts.zip`.
4. Descompriman `artifacts.zip` dentro de su carpeta `fuelwise/` local y corran `streamlit run app/app.py`.
   (La última celda opcional también abre la app desde Colab con un enlace temporal de Cloudflare.)

---

## 3. Estructura

```
fuelwise/
├── config.py                 # constantes, hiperparámetros y supuestos económicos (con fuentes)
├── run_all.py                # orquesta todo el pipeline
├── src/
│   ├── download_data.py      # descarga y descomprime VED
│   ├── preprocess.py         # filtro gasolina, Algoritmo 1 VED, pasos de 10 s, calidad, split
│   ├── windows.py            # escalado (solo train) y ventanas sin fuga de información
│   ├── models.py             # LSTM y GRU
│   ├── train.py              # entrenamiento con early stopping
│   ├── evaluate.py           # simulación por bloques, baselines, métricas, tiempo de inferencia
│   ├── value_model.py        # función de valor, margen óptimo y curva valor-vs-error
│   └── inference.py          # pronóstico de un trayecto individual (app)
├── app/app.py                # aplicación Streamlit
├── app/sample_trip.csv       # trayecto crudo de ejemplo (formato VED) para probar la carga de CSV
├── notebooks/FuelWise_Colab.ipynb
└── artifacts/                # modelos, métricas y predicciones (lo que usa la app)
```

---

## 4. Metodología

### 4.1 Datos y verificación de mediciones de consumo
- 264 vehículos ICE en los datos estáticos; 13.1 M registros OBD-II de esos vehículos.
- **Hallazgo clave:** `Fuel Rate[L/hr]` está presente en solo el **0.01 %** de los registros ICE (1 vehículo,
  que además es diésel). El paper de VED (Oh et al., 2022, Sección II.C) lo reporta: *"For most ICE vehicles
  and HEVs, FuelRate ... is not available"* y propone el **Algoritmo 1**, que implementamos tal cual:

  ```
  correction = (1 + STFT/100 + LTFT/100) / AFR          AFR = 14.08 (E10)
  si hay Fuel Rate → se usa
  si no, si hay MAF → combustible[g/s] = MAF · correction
  si no → MAF = Load/100 · ρ_aire · cilindrada · RPM / 120 ;  combustible = MAF · correction
  FCR[L/h] = combustible[g/s] · 3600 / 745 g/L
  ```
  Fuentes del consumo en registros ICE: 10.40 M por MAF+trims, 2.09 M por carga·RPM·cilindrada, 0.64 M sin dato.
- Se excluye el único ICE diésel (VehId 119).
- Agregación a pasos de **10 s**; huecos ≤ 30 s se interpolan dentro del trayecto.
- Filtro de calidad: trayectos ≥ 3 min y con ≥ 90 % de pasos con velocidad, RPM, carga y consumo válidos.
- Resultado: **210 vehículos, 12 058 trayectos, 2 005 h de conducción, 7 659 L**.

### 4.2 División sin fuga de información
- División **por vehículo** (70/15/15 → 147/32/31 vehículos; 8 884/1 330/1 844 trayectos), con trayectos
  completos y **antes** de construir ventanas: ninguna ventana cruza conjuntos y los vehículos de prueba
  nunca se ven en entrenamiento.
- La estandarización (media/desviación) se ajusta solo con entrenamiento.

### 4.3 Pronóstico
- Cada **2 min** se emite un pronóstico del consumo medio (L/h) de los **próximos 2 min**, usando solo los
  **últimos 3 min** (18 pasos) de velocidad, RPM, carga y consumo. Nunca se usan valores futuros.
  Al inicio del trayecto la historia faltante se rellena con ceros y una máscara `is_pad`.
- Litros del trayecto = Σ (pronóstico L/h × duración del bloque). Costo = litros × precio.
- Modelos: LSTM y GRU de 2 capas, 64 unidades, dropout 0.2, cabeza densa 64→32→1; Adam (lr 1e-3),
  MSE, batch 1 024, early stopping (paciencia 5) con MAE de validación, máx. 30 épocas.
- Referencias sin deep learning:
  - **Promedio histórico del vehículo**: L/h de sus trayectos anteriores (orden cronológico);
    si no hay historial, promedio de entrenamiento.
  - **Persistencia**: consumo medio de los últimos 3 min del mismo trayecto.

### 4.4 Iteraciones (`artifacts/iterations.csv`)

| Iter | Cambio | LSTM MAE (L/h) | GRU MAE (L/h) | LSTM / GRU error mediano por trayecto |
|---|---|---|---|---|
| 1 | Predicción directa del consumo | 1.154 | 1.144 | 9.5 % / 9.6 % |
| **2** ✔ | **Residual: aprende la corrección sobre la persistencia** | **1.154** | **1.144** | **9.7 % / 9.1 %** |
| 3 | + promedio acumulado del consumo del trayecto como entrada | 1.143 | 1.139 | 10.0 % / 10.0 % |

(Valores en prueba.) La iteración 2 se seleccionó por su menor error por trayecto **en validación**
(LSTM 9.9 %, GRU 9.5 %), que es la métrica que alimenta la decisión de reserva, y porque usa
exactamente las variables de la propuesta. Los modelos de las otras iteraciones están en `artifacts/iter*/`.

---

## 5. Resultados (conjunto de prueba: 31 vehículos no vistos, 1 844 trayectos, 10 172 pronósticos)

### Pronóstico de 2 min (L/h)

| Método | MAE | RMSE | Sesgo |
|---|---|---|---|
| **GRU** | **1.144** | **1.613** | −0.150 |
| LSTM | 1.154 | 1.632 | −0.185 |
| Promedio histórico | 1.329 | 1.819 | +0.010 |
| Persistencia | 1.395 | 1.914 | +0.014 |

GRU reduce el MAE **14 %** frente al promedio histórico y **18 %** frente a la persistencia.

### Litros por trayecto (integrando los pronósticos)

| Método | MAE (L) | Error mediano | Error total de la flota |
|---|---|---|---|
| Persistencia | **0.035** | **4.8 %** | +0.8 % |
| GRU | 0.072 | 9.1 % | −3.1 % |
| LSTM | 0.078 | 9.7 % | −4.0 % |
| Promedio histórico | 0.134 | 14.1 % | +0.6 % |

**Interpretación honesta:** las redes son mejores pronosticando los próximos 2 minutos, pero al
*integrar* pronósticos para el total del trayecto la persistencia gana. Razón: la persistencia es la
señal real retrasada; al sumar sus bloques, cada paso real cuenta prácticamente una vez y los errores se
cancelan. Las RNN tienden hacia el consumo promedio de la flota y tienen un sesgo negativo leve en
vehículos nuevos, y esos errores no se cancelan. Frente al método de la propuesta (promedio histórico), las
RNN sí mejoran: el error mediano baja de 14.1 % a 9.1 %.

### Costo computacional (CPU)

| Modelo | Parámetros | MFLOPs/pronóstico | Latencia (lote 1) | Entrenamiento |
|---|---|---|---|---|
| LSTM | 53 569 | 1.84 | 0.34 ms | 230 s (15 épocas) |
| GRU | 40 705 | 1.38 | 0.76 ms | 319 s (23 épocas) |

### Impacto económico (margen elegido en validación, resultado en prueba)

Supuestos por defecto (ajustables en la app): precio 1.128 USD/L; ahorro por acierto 100 USD; costo fijo de
déficit 35 USD; costo de exceso 10 % del precio por litro sobrante.

| Método | Margen | Cobertura | Emergencias | Sobrante/1 000 trayectos | Δ valor vs histórico / 1 000 trayectos |
|---|---|---|---|---|---|
| Persistencia | +94 % | 100 % | 0 | 654 L | +148 USD |
| GRU | +145 % | 100 % | 0 | 941 L | +116 USD |
| LSTM | +160 % | 99.95 % | 1 | 1 024 L | +33 USD |
| Promedio histórico | +285 % | 100 % | 0 | 1 968 L | 0 |

Con cobertura objetivo de 95 %, el margen necesario es +58 % con el promedio histórico, +29 % con GRU y
+16 % con persistencia; el sobrante cae de 411 L a 174 L (GRU) por cada 1 000 trayectos.
Como un déficit cuesta mucho más que un litro sobrante, el margen óptimo es alto y la diferencia en
dólares por trayecto es pequeña. **Lo que mejora con una predicción más precisa es cuánto combustible
hay que inmovilizar para lograr la misma cobertura.** No afirmamos ahorros verificados: son resultados
bajo supuestos documentados.

---

## 6. Supuestos y fuentes

| Supuesto | Valor | Fuente |
|---|---|---|
| AFR estequiométrico E10 | 14.08 | Oh et al. (2022), VED, Algoritmo 1 |
| Densidad de gasolina | 745 g/L | Supuesto (rango típico 0.72–0.77 kg/L) |
| Densidad del aire (respaldo sin MAF) | 1.225 g/L | Atmósfera estándar |
| Precio del combustible | 4.269 USD/gal = 1.128 USD/L | EIA, *Gasoline and Diesel Fuel Update*, regular Midwest (PADD 2), 26-oct-2026 |
| Ahorro por acierto | 100 USD | Engineer Fix (dic-2025): cargo base no-miembro por falta de combustible 50–150 USD (punto medio) |
| Costo fijo de déficit | 35 USD | Engineer Fix: recargos nocturnos/fin de semana 20–50 USD (punto medio) |
| Prima por litro faltante | 0 | Engineer Fix: el combustible entregado se cobra a precio de mercado |
| Costo de exceso | 10 % del precio por litro | Supuesto del equipo (capital inmovilizado) |

## 7. Limitaciones
- El "consumo real" es una estimación física (MAF y fuel trims), no una medición directa del flujo de combustible.
- Datos de Ann Arbor (2017–2018): clima, tráfico y vehículos distintos a otros contextos; trayectos mayormente urbanos y cortos.
- La integración de pronósticos usa la duración real del trayecto; no predice la duración.
- No aplica a híbridos, eléctricos ni diésel.

## Referencias
- Oh, G. S., LeBlanc, D. J., & Peng, H. (2022). Vehicle Energy Dataset (VED). *IEEE T-ITS*, 23(4), 3302–3312. https://doi.org/10.1109/TITS.2020.3035596
- Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural Computation*, 9(8), 1735–1780.
- Cho, K. et al. (2014). Learning phrase representations using RNN encoder–decoder. *EMNLP*, 1724–1734.
- U.S. EIA. Gasoline and Diesel Fuel Update. https://www.eia.gov/petroleum/gasdiesel/
- Engineer Fix. How Much Is Roadside Assistance for Gas? https://engineerfix.com/how-much-is-roadside-assistance-for-gas/
