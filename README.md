# FuelWise AI

Asistente de combustible para flotas. Claude Haiku 5.5 responde en español preguntas como cuánto reservar o si alcanza una cantidad. Cada cifra sale de herramientas que consultan una red GRU entrenada con datos OBD-II, así que el LLM no inventa números.

CC3092 Deep Learning y Sistemas Inteligentes, sección 20, Universidad del Valle de Guatemala.

| Integrante | Carné |
|---|---|
| Mia Fuentes | 23775 |
| Roberto Barreda | 23354 |

## Ejecutar la app

```bash
git clone https://github.com/miafuentes30/CC3092-PRY3.git
cd CC3092-PRY3
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app/app.py
```

En macOS o Linux, activar con `source .venv/bin/activate`. La app abre en `http://localhost:8501` y no necesita el dataset ni GPU.

Para el asistente en vivo, copiar `.env.example` como `.env` y poner la llave de Anthropic. Sin llave muestra respuestas guardadas.

## Reproducir las métricas

| Comando | Qué hace |
|---|---|
| `python -m src.llm_eval` | Evalúa el asistente con 40 preguntas. Requiere la llave |
| `python -m src.llm_eval --configs reglas --no-demo` | Evalúa solo el baseline sin LLM |
| `python -m src.llm_eval --configs qwen_v2 --no-demo` | Evalúa el modelo abierto. Requiere GPU, `transformers` y unos 3 GB de pesos |
| `python -m src.llm_eval --carga` | Prueba de carga: 20 preguntas con 5 simultáneas |
| `python run_all.py` | Descarga VED, preprocesa, entrena y evalúa el GRU. Unos 20 min en CPU |

## Resultados

Asistente, 40 preguntas:

| Configuración | Exactitud | Respuestas con cifras sin respaldo | Costo por consulta |
|---|---|---|---|
| Reglas sin LLM | 67.5 % | 0 % | 0 |
| Haiku sin herramientas | 5.0 % | 60 % | 0.0002 USD |
| Qwen2.5 1.5B abierto, local, con herramientas | 40.0 % | 15 % | 0, corre en GPU |
| **Haiku con herramientas, final** | **97.5 %** | **0 %** | **0.0004 USD** |

Modelo de pronóstico que consultan las herramientas, en 31 vehículos no vistos:

| Método | MAE L/h | Error mediano por trayecto |
|---|---|---|
| **GRU** | **1.144** | **9.1 %** |
| Promedio histórico | 1.329 | 14.1 % |

La revisión manual de las 40 respuestas está en `eval/revision_manual.csv`. La metodología, las iteraciones, las fallas y las limitaciones están en `docs/FuelWise_Reporte.docx`, también exportado como PDF.

## Datos

Vehicle Energy Dataset, Oh, LeBlanc y Peng, IEEE T-ITS 2022. https://github.com/gsoh/VED, licencia Apache 2.0.
