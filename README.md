# FuelWise AI

Aplicación web que pronostica el consumo de combustible de un vehículo de gasolina cada 2 minutos con una red GRU. Un asistente con Claude Haiku 5.5 responde en español preguntas de la flota, como cuánto reservar, y toma cada cifra del modelo.

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
| `python run_all.py` | Descarga VED, preprocesa, entrena y evalúa el GRU. Unos 20 min en CPU |
| `python -m src.llm_eval` | Evalúa el asistente con 40 preguntas. Requiere la llave |
| `python -m src.llm_eval --configs reglas --no-demo` | Evalúa solo el baseline sin LLM |

## Resultados

Pronóstico a 2 minutos en 31 vehículos no vistos:

| Método | MAE L/h | Error mediano por trayecto |
|---|---|---|
| **GRU** | **1.144** | **9.1 %** |
| Promedio histórico | 1.329 | 14.1 % |

Asistente, 40 preguntas:

| Configuración | Exactitud | Respuestas con cifras sin respaldo | Costo por consulta |
|---|---|---|---|
| Reglas sin LLM | 67.5 % | 0 % | 0 |
| Haiku sin herramientas | 7.5 % | 60 % | 0.0002 USD |
| **Haiku con herramientas, final** | **97.5 %** | **0 %** | **0.0004 USD** |

La metodología, las iteraciones, las fallas y las limitaciones están en `docs/FuelWise_Reporte.docx`, también exportado como PDF.

## Datos

Vehicle Energy Dataset, Oh, LeBlanc y Peng, IEEE T-ITS 2022. https://github.com/gsoh/VED, licencia Apache 2.0.
