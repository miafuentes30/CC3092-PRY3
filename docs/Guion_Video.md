# Guion del video de demostración

FuelWise AI. Duración objetivo: 2 min 30 s. Mia Fuentes, 23775. Roberto Barreda, 23354.

## Antes de grabar

1. Poner la llave en `.env` y abrir la app con `streamlit run app/app.py`.
2. Dejarla en la pestaña Trayecto: vehículo 157, trayecto 2898, minuto 6.
3. Grabar a 1920 x 1080 con el navegador en pantalla completa, F11, y el zoom en 110 %.
4. Cerrar notificaciones y otras pestañas.
5. Grabar con Win + Alt + R, la barra de juegos de Windows, o con OBS. Exportar en .mp4.
6. Leer la narración una vez en voz alta antes de grabar. Son unas 260 palabras; con las pausas de la demo caben en 2 min 30 s.

## Escenas

### 1. Presentación, 0:00 a 0:15

**Pantalla.** La app en la pestaña Trayecto, sin moverla.

**Narración.** FuelWise pronostica cada 2 minutos el consumo de combustible de un vehículo de gasolina y responde en español las preguntas de un administrador de flota. Lo hicimos Mia Fuentes y Roberto Barreda para el curso de Deep Learning.

### 2. El pronóstico, 0:15 a 0:50

**Pantalla.** Señalar el selector de vehículo. Mover el control de minuto de 6 a 12. Señalar la línea gris punteada, la franja sombreada y el número grande de la derecha.

**Narración.** Este vehículo no se usó para entrenar. Cada 2 minutos, una red GRU mira solo los últimos 3 minutos de velocidad, RPM, carga y consumo, y pronostica los 2 minutos siguientes. La línea punteada es lo que todavía no ocurre. La franja es el rango probable, calculado con los errores de validación. A la derecha están el pronóstico, lo medido hasta ahora y el costo.

### 3. El asistente, 0:50 a 1:35

**Pantalla.** Presionar "¿Me alcanza con un litro para 20 minutos más?". Abrir "De dónde salen las cifras". Después escribir en el campo de texto: `cuanto llevo gastado y cuanto reservo pa 10 min`.

**Narración.** El administrador puede preguntar con sus palabras. Claude Haiku interpreta la pregunta, pero no calcula nada: llama una herramienta que consulta el modelo y redacta con esas cifras. Aquí se ve qué herramienta usó, con qué datos, cuánto tardó y cuánto costó. También entiende lenguaje coloquial y dos preguntas en una.

### 4. Un trayecto propio, 1:35 a 1:55

**Pantalla.** Pestaña Subir trayecto. En Ejemplo, elegir "Carretera, 12 min a 89 km/h". Mover el control de minuto.

**Narración.** La app también acepta un trayecto propio en el formato del Vehicle Energy Dataset. Aquí, un tramo de carretera de 12 minutos con el pronóstico del GRU contra el promedio histórico.

### 5. La evidencia, 1:55 a 2:20

**Pantalla.** Pestaña Modelo: dejar visible la primera frase y la tabla. Pestaña Asistente: dejar visible la gráfica de barras.

**Narración.** En 31 vehículos que el modelo no vio, el GRU tiene 14 % menos error que el promedio histórico. Evaluamos el asistente con 40 preguntas: acierta 97.5 % y no dio cifras sin respaldo. Las reglas por palabras clave aciertan solo 35 % de las preguntas escritas con lenguaje natural.

### 6. Cierre, 2:20 a 2:30

**Pantalla.** Volver a la pestaña Trayecto.

**Narración.** La red pronostica, el LLM traduce a lenguaje natural y cada cifra se puede rastrear hasta el modelo. El código y la evaluación están en GitHub.

## Si algo falla al grabar

1. Si el asistente tarda, esperar sin cortar; en el video final se puede recortar la espera.
2. Si no hay internet, los cinco botones sugeridos responden con respuestas guardadas en el minuto 6 del trayecto por defecto. En ese caso, saltar la pregunta escrita de la escena 3.
