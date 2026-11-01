# FuelWise AI

Aplicación web que pronostica el consumo de combustible de un vehículo de gasolina cada 2 minutos durante un trayecto. Una red GRU hace el pronóstico y un asistente con Claude Haiku 5.5 responde en español preguntas del administrador de flota, como cuánto reservar o si alcanza una cantidad. El asistente toma cada cifra de herramientas que consultan el modelo, así que no inventa números.

CC3092 Deep Learning y Sistemas Inteligentes, sección 20, Universidad del Valle de Guatemala.

| Integrante | Carné |
|---|---|
| Mia Fuentes | 23775 |
| Roberto Barreda | 23354 |

## Ejecutar la app

Los modelos entrenados están en `artifacts/`. La app no necesita el dataset ni GPU.

```bash
git clone https://github.com/miafuentes30/CC3092-PRY3.git
cd CC3092-PRY3
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app/app.py
```

En macOS o Linux, activar el entorno con `source .venv/bin/activate`. La app abre en `http://localhost:8501`.

Para usar el asistente en vivo, copiar `.env.example` como `.env` y poner la llave de Anthropic. Sin llave, el asistente muestra respuestas guardadas para las preguntas sugeridas del trayecto que la app abre por defecto.

## Reproducir las métricas

```bash
python run_all.py
```

Descarga VED, unos 176 MB, y luego preprocesa, entrena, evalúa y calcula el modelo de valor. En CPU tarda unos 20 minutos. `python run_all.py --quick` hace una prueba corta. Para usar GPU en Colab está `notebooks/FuelWise.ipynb`.

Evaluación del asistente, con la llave en `.env`:

```bash
python -m src.llm_eval
```

Costó unos 0.45 USD, casi todo por Sonnet. `python -m src.llm_eval --configs reglas --no-demo` corre solo el baseline sin LLM, que no necesita llave.

## Estructura

| Archivo | Función |
|---|---|
| `src/download_data.py` | Descarga VED |
| `src/preprocess.py` | Filtra gasolina, estima el consumo, agrega a pasos de 10 s, controla calidad y divide por vehículo |
| `src/windows.py` | Escala con datos de entrenamiento y arma ventanas sin datos futuros |
| `src/models.py`, `src/train.py` | LSTM, GRU y entrenamiento con early stopping |
| `src/evaluate.py` | Simula el uso por bloques y calcula métricas contra las referencias |
| `src/value_model.py` | Función de valor y margen de reserva |
| `src/assistant.py` | Asistente: herramientas, prompts versionados, llamada a Claude y baseline de reglas |
| `src/llm_eval.py` | Evaluación del asistente y caché para usar la app sin conexión |
| `eval/llm_eval_set.json` | 40 preguntas de evaluación del asistente |
| `app/app.py` | Aplicación Streamlit |
| `config.py` | Hiperparámetros y supuestos económicos con su fuente |
| `artifacts/` | Modelos, métricas e historial de iteraciones |

## Metodología

1. **Datos.** Vehicle Energy Dataset, 264 vehículos de gasolina en Ann Arbor, 2017 a 2018. Solo el 0.01 % de los registros trae el flujo de combustible, así que el consumo se estima con el Algoritmo 1 del artículo de VED a partir de MAF y fuel trims.
2. **Limpieza.** Pasos de 10 s, huecos de hasta 30 s interpolados, trayectos de 3 min o más con 90 % de datos válidos. Quedan 210 vehículos, 12 058 trayectos y 2 005 horas.
3. **División por vehículo.** 70 % entrenamiento, 15 % validación y 15 % prueba. Los vehículos de prueba no se ven al entrenar y el escalado se ajusta solo con entrenamiento.
4. **Pronóstico.** Cada 2 min el modelo recibe los últimos 3 min de velocidad, RPM, carga y consumo, y predice el consumo medio de los 2 min siguientes. No usa datos futuros.
5. **Modelos.** LSTM y GRU de 2 capas y 64 unidades que aprenden la corrección sobre la persistencia. Adam, pérdida MSE y early stopping con el MAE de validación.
6. **Referencias.** Promedio histórico del vehículo, que es como se decide hoy, y persistencia, que repite el consumo de los últimos 3 min.

## Asistente con LLM

1. **Por qué un LLM.** El administrador pregunta con sus palabras, por ejemplo "¿me alcanza con un litro para 20 minutos?". Un enrutador por palabras clave falla cuando la frase cambia. El LLM interpreta la pregunta y decide qué consultar, y las cifras siguen saliendo del modelo GRU.
2. **Modelo.** Claude Haiku 5.5, elegido por costo y latencia, con esfuerzo bajo. Claude Sonnet 5.5 se evaluó como comparación. Ambos son modelos cerrados que se usan vía API.
3. **Técnica.** Prompting zero-shot con uso de herramientas. Hay cuatro herramientas deterministas: estado del trayecto, pronóstico a 2 min con rango P10 a P90, reserva con margen calibrado en validación y métricas del modelo. Es transfer learning a escala: el modelo preentrenado no se ajusta, solo recibe instrucciones y herramientas.
4. **Prompt final.** `v2_reglas` en `src/assistant.py`. Pide tomar cada cifra de una herramienta, responder en 2 a 4 oraciones, rechazar lo que no trata del trayecto y no seguir instrucciones escritas en la pregunta. Los modelos actuales de Claude no aceptan temperature ni top_p. La generación se controla con un máximo de 4 000 tokens y esfuerzo bajo.
5. **Control del contexto.** Una consulta típica usa unos 3 300 tokens de entrada y 200 de salida, en dos llamadas. La pregunta se corta a 500 caracteres y el bucle de herramientas a 5 rondas. Cada pregunta se responde por separado, sin historial.
6. **Evaluación.** 40 preguntas sobre 8 trayectos de prueba. 20 son directas y usan el vocabulario de la app. Las otras 20 son variadas: coloquiales, con errores de escritura, en inglés, con dos intenciones o pidiendo una explicación. La referencia de cada caso se calcula con las mismas herramientas. Se mide exactitud, respuestas con cifras no trazables, latencia, tokens y costo, contra dos baselines: reglas por palabras clave sin LLM y Haiku sin herramientas.

### Resultados del asistente

| Configuración | Exactitud | Directas | Variadas | Respuestas con cifras inventadas | Latencia media | Costo por consulta |
|---|---|---|---|---|---|---|
| Reglas sin LLM | 67.5 % | 100 % | 35 % | 0 % | 0.005 s | 0 |
| Haiku sin herramientas | 7.5 % | 10 % | 5 % | 65 % | 2.6 s | 0.0002 USD |
| Haiku con prompt mínimo | 82.5 % | 85 % | 80 % | 17.5 % | 2.8 s | 0.0004 USD |
| **Haiku con reglas, final** | **97.5 %** | **100 %** | **95 %** | **0 %** | **2.2 s** | **0.0004 USD** |
| Sonnet 5.5 con reglas | 100 % | 100 % | 100 % | 0 % | 2.9 s | 0.0083 USD |

1. Sin herramientas, Haiku no tiene los datos: se niega o inventa cifras en el 65 % de las respuestas.
2. Las reglas por palabras clave aciertan todas las preguntas directas, pero solo el 35 % de las variadas. Se escribieron con las preguntas directas a la vista, así que ese 100 % es un techo optimista.
3. El prompt final elimina las cifras inventadas. Con el prompt mínimo, Haiku calculaba por su cuenta datos como litros por 100 km o minutos de autonomía.
4. Sonnet acierta una pregunta más, pero cuesta 19 veces más y tarda 0.7 s más. La app usa Haiku.
5. La única falla del modelo final fue una pregunta con dos intenciones: respondió el pronóstico y omitió los litros consumidos.
6. La primera versión del calificador marcaba como fallas respuestas correctas, por ejemplo al leer "1,844 trayectos" como 1.844 litros. Se corrigió, se revisaron a mano los casos que cambiaron y se volvió a calificar con `python -m src.llm_eval --regrade`, sin llamar otra vez al modelo. El detalle por caso está en `artifacts/llm/`.

## Resultados del pronóstico en prueba

31 vehículos no vistos, 1 844 trayectos y 10 172 pronósticos.

| Método | MAE L/h | RMSE L/h | Error mediano por trayecto |
|---|---|---|---|
| **GRU** | **1.144** | **1.613** | 9.1 % |
| LSTM | 1.154 | 1.632 | 9.7 % |
| Promedio histórico | 1.329 | 1.819 | 14.1 % |
| Persistencia | 1.395 | 1.914 | 4.8 % |

GRU reduce el MAE 14 % frente al promedio histórico y 18 % frente a la persistencia. Cada pronóstico tarda menos de 1 ms en CPU.

En el total del trayecto gana la persistencia. Al sumar sus bloques reproduce el consumo real con 3 min de retraso, y sus errores se cancelan. Frente al promedio histórico, GRU baja el error mediano de 14.1 % a 9.1 %.

**Reserva.** Para cubrir el 95 % de los trayectos, el promedio histórico necesita reservar 58 % extra y GRU 29 %. El combustible sobrante baja de 411 L a 174 L por cada 1 000 trayectos. Los supuestos económicos están en `config.py` y se pueden cambiar en la app.

## Iteraciones

La selección se hizo con el error por trayecto en validación. El registro completo está en `artifacts/iterations.csv`.

| Iteración | Cambio | MAE GRU en prueba | Error GRU por trayecto en validación |
|---|---|---|---|
| 1 | Predicción directa del consumo | 1.144 | 9.7 % |
| **2, final** | Corrección sobre la persistencia | 1.144 | **9.5 %** |
| 3 | Agrega el consumo promedio acumulado del trayecto | 1.139 | 9.9 % |

## Fallas observadas

1. Al inicio del trayecto no hay historia y las redes quedan detrás del promedio histórico, con MAE de 1.39 frente a 1.35.
2. En el 5 % de bloques con mayor consumo, GRU subestima 3.8 L/h en promedio.
3. El 68 % de los vehículos de prueba tiene sesgo negativo. El modelo tiende al promedio de la flota.

## Limitaciones

1. El consumo real es una estimación física, no una medición directa.
2. Datos de una sola ciudad con trayectos cortos, de 10 min en promedio.
3. No aplica a híbridos, eléctricos ni diésel.
4. Los parámetros económicos son supuestos documentados.
5. El asistente necesita internet y la API de Anthropic. Sin conexión solo muestra respuestas guardadas.
6. Al asistente solo se envían la pregunta y cifras del trayecto, sin datos personales. No debe usarse para decisiones de seguridad del vehículo.

## Referencias

1. Oh, G. S., LeBlanc, D. J., y Peng, H. 2022. Vehicle Energy Dataset, a large-scale dataset for vehicle energy consumption research. IEEE Transactions on Intelligent Transportation Systems, 23, 3302 a 3312. https://doi.org/10.1109/TITS.2020.3035596
2. Hochreiter, S., y Schmidhuber, J. 1997. Long short-term memory. Neural Computation, 9, 1735 a 1780.
3. Cho, K. et al. 2014. Learning phrase representations using RNN encoder-decoder for statistical machine translation. EMNLP, 1724 a 1734.
4. U.S. EIA. Gasoline and Diesel Fuel Update. https://www.eia.gov/petroleum/gasdiesel/
5. Engineer Fix. How Much Is Roadside Assistance for Gas? https://engineerfix.com/how-much-is-roadside-assistance-for-gas/
6. Datos: https://github.com/gsoh/VED, licencia Apache 2.0.
