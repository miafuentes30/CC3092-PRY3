"""FuelWise AI: aplicación web en Streamlit.

Ejecutar desde la carpeta raíz del proyecto:
    streamlit run app/app.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config as C  # noqa: E402
from src import assistant  # noqa: E402
from src.value_model import (ValueParams, error_value_curve, log_error_sigma,  # noqa: E402
                             outcome, value_curve)

st.set_page_config(page_title="FuelWise AI", page_icon="⛽", layout="wide")

LABELS = {"lstm": "LSTM", "gru": "GRU", "hist": "Promedio histórico", "persist": "Persistencia",
          "real": "Real"}
COLORS = {"real": "#1f2937", "lstm": "#2563eb", "gru": "#16a34a", "hist": "#d97706",
          "persist": "#9333ea"}
STEP_MIN = C.STEP_S / 60.0
BLOCK_MIN = C.HORIZON * C.STEP_S / 60.0


# --------------------------------------------------------------------------
# Carga de artefactos
# --------------------------------------------------------------------------
@st.cache_data
def load_json(name):
    p = C.ART_DIR / name
    return json.loads(p.read_text()) if p.exists() else None


@st.cache_data
def load_pq(name):
    p = C.ART_DIR / name
    return pd.read_parquet(p) if p.exists() else None


@st.cache_resource
def bundle():
    from src.inference import load_bundle
    return load_bundle()


metrics = load_json("metrics.json")
if metrics is None:
    st.error("No se encontraron artefactos entrenados en `artifacts/`. Ejecuta primero "
             "`python run_all.py` (o descomprime los artefactos preentrenados).")
    st.stop()

train_sum = load_json("training_summary.json")
quality = json.loads((C.PROC_DIR / "quality_report.json").read_text()) \
    if (C.PROC_DIR / "quality_report.json").exists() else load_json("quality_report.json")
blocks_test, blocks_val = load_pq("blocks_test.parquet"), load_pq("blocks_val.parquet")
trips_test, trips_val = load_pq("trips_test.parquet"), load_pq("trips_val.parquet")
test_bins = load_pq("test_bins.parquet")
METHODS = [m for m in ["lstm", "gru", "hist", "persist"] if m in blocks_test]


@st.cache_data
def residual_quantiles():
    """Intervalo empírico (P10–P90) del error por bloque en VALIDACIÓN."""
    return {m: (float(np.quantile(blocks_val["real"] - blocks_val[m], 0.10)),
                float(np.quantile(blocks_val["real"] - blocks_val[m], 0.90))) for m in METHODS}


RQ = residual_quantiles()

# --------------------------------------------------------------------------
# Barra lateral
# --------------------------------------------------------------------------
st.sidebar.title("⛽ FuelWise AI")
st.sidebar.caption("Pronóstico de consumo de combustible con LSTM y GRU sobre el Vehicle Energy Dataset.")
model_sel = st.sidebar.selectbox("Modelo principal", ["gru", "lstm"], format_func=LABELS.get)
baseline_sel = st.sidebar.selectbox("Método de referencia", ["hist", "persist"], format_func=LABELS.get)
st.sidebar.markdown("### Supuestos económicos")
currency = st.sidebar.text_input("Moneda", "USD")
price = st.sidebar.number_input(f"Precio del combustible ({currency}/L)", 0.01, 100.0,
                                float(C.FUEL_PRICE_USD_L), 0.01,
                                help="Por defecto: EIA, gasolina regular Midwest, 26-oct-2026 "
                                     "(4.269 USD/gal).")
with st.sidebar.expander("Parámetros del modelo de valor"):
    saving_hit = st.number_input("Ahorro por acierto (emergencia evitada)", 0.0, 10000.0,
                                 C.SAVING_PER_HIT_USD, 5.0)
    deficit_fixed = st.number_input("Costo fijo de déficit (recargo)", 0.0, 10000.0,
                                    C.DEFICIT_FIXED_USD, 5.0)
    deficit_premium = st.slider("Prima por litro faltante (× precio)", 0.0, 3.0,
                                C.DEFICIT_PREMIUM_FRAC, 0.05)
    excess_frac = st.slider("Costo por litro en exceso (× precio)", 0.0, 1.0, C.EXCESS_COST_FRAC, 0.01)
PRM = ValueParams(price, saving_hit, deficit_fixed, deficit_premium, excess_frac)


def money(x):
    return f"{x:,.2f} {currency}"


tabs = st.tabs(["🚗 Simulador de trayecto", "💬 Asistente", "📈 Real vs predicho", "🧠 Rendimiento de modelos",
                "💰 Impacto económico", "📂 Probar con tu CSV", "ℹ️ Datos y metodología"])


# --------------------------------------------------------------------------
# Gráfica de un trayecto (compartida por el simulador y el CSV propio)
# --------------------------------------------------------------------------
def trip_figure(bins, blk, now_block, show):
    fig = go.Figure()
    tmin = bins["step"].to_numpy() * STEP_MIN
    now_min = now_block * BLOCK_MIN
    seen = tmin < now_min
    fig.add_trace(go.Scatter(x=tmin[seen], y=bins["fcr"].to_numpy()[seen], name="Consumo observado",
                             line=dict(color=COLORS["real"], width=1.5)))
    fig.add_trace(go.Scatter(x=tmin[~seen], y=bins["fcr"].to_numpy()[~seen],
                             name="Consumo futuro (no visible para el modelo)",
                             line=dict(color="#9ca3af", width=1, dash="dot")))
    for m in show:
        xs, ys = [], []
        for _, r in blk[blk["t"] <= now_block * C.HORIZON].iterrows():
            x0 = r["t"] * STEP_MIN
            xs += [x0, x0 + r["block_len"] * STEP_MIN, None]
            ys += [r[m], r[m], None]
        fig.add_trace(go.Scatter(x=xs, y=ys, name=f"Pronóstico {LABELS[m]}", mode="lines",
                                 line=dict(color=COLORS[m], width=3)))
    cur = blk[blk["t"] == now_block * C.HORIZON]
    if len(cur):
        r = cur.iloc[0]
        lo, hi = RQ[show[0]]
        x0, x1 = r["t"] * STEP_MIN, (r["t"] + r["block_len"]) * STEP_MIN
        fig.add_trace(go.Scatter(x=[x0, x1, x1, x0], y=[max(r[show[0]] + lo, 0)] * 2 + [r[show[0]] + hi] * 2,
                                 fill="toself", fillcolor="rgba(37,99,235,0.15)", line=dict(width=0),
                                 name="Rango probable P10–P90", hoverinfo="skip"))
    fig.add_vline(x=now_min, line=dict(color="#ef4444", dash="dash"), annotation_text="ahora")
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10), xaxis_title="Minuto del trayecto",
                      yaxis_title="Consumo (L/h)", legend=dict(orientation="h", y=-0.25))
    return fig


def cumulative_figure(blk, show):
    fig = go.Figure()
    x = (blk["t"] + blk["block_len"]) * STEP_MIN
    for m in ["real"] + show:
        fig.add_trace(go.Scatter(x=x, y=blk[f"L_{m}"].cumsum(), name=LABELS[m],
                                 line=dict(color=COLORS[m], width=3 if m == "real" else 2,
                                           dash=None if m == "real" else "dash")))
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=30, b=10), xaxis_title="Minuto",
                      yaxis_title="Litros acumulados", legend=dict(orientation="h", y=-0.3))
    return fig


# --------------------------------------------------------------------------
# 1. Simulador
# --------------------------------------------------------------------------
with tabs[0]:
    st.header("Simulador de trayecto en tiempo real")
    st.write(f"Trayectos de **vehículos que el modelo nunca vio** (conjunto de prueba). Cada "
             f"{BLOCK_MIN:.0f} min el sistema pronostica el consumo (L/h) de los siguientes "
             f"{BLOCK_MIN:.0f} min usando solo los últimos {C.LOOKBACK*C.STEP_S//60} min de "
             "velocidad, RPM, carga y consumo. Mueve el control para avanzar en el tiempo.")
    tt = trips_test.copy()
    c1, c2 = st.columns(2)
    veh_opts = sorted(tt["veh"].unique())
    default_veh = int(tt.loc[(tt["duration_min"] - 20).abs().idxmin(), "veh"])
    veh = c1.selectbox("Vehículo", veh_opts, index=veh_opts.index(default_veh),
                       format_func=lambda v: f"Vehículo {v}, {tt.loc[tt.veh == v, 'engine'].iloc[0]}"
                       + ("" if tt.loc[tt.veh == v, 'vclass'].iloc[0] == "NO DATA"
                          else f" ({tt.loc[tt.veh == v, 'vclass'].iloc[0]})"))
    tv = tt[tt["veh"] == veh].sort_values("duration_min", ascending=False)
    demo_veh, demo_trip, _ = assistant.demo_trip()
    trip = c2.selectbox("Trayecto", tv["trip"].tolist(),
                        index=tv["trip"].tolist().index(demo_trip) if veh == demo_veh else 0,
                        format_func=lambda t: (lambda r: f"Trip {t} · {r.duration_min:.0f} min · "
                                               f"{r.distance_km:.1f} km")(tv[tv.trip == t].iloc[0]))
    bins = test_bins[(test_bins.veh == veh) & (test_bins.trip == trip)].sort_values("step")
    blk = blocks_test[(blocks_test.veh == veh) & (blocks_test.trip == trip)].sort_values("t").reset_index(drop=True)
    n_blocks = len(blk)
    now_block = st.slider("Bloque actual (pronóstico emitido)", 0, max(n_blocks - 1, 0), min(3, n_blocks - 1),
                          help=f"Cada bloque dura {BLOCK_MIN:.0f} min.")
    show = [model_sel, "gru" if model_sel == "lstm" else "lstm", baseline_sel]
    st.plotly_chart(trip_figure(bins, blk, now_block, show), width="stretch")

    r = blk.iloc[now_block]
    so_far = blk.iloc[:now_block + 1]
    k = st.columns(4)
    k[0].metric(f"Pronóstico próximos {BLOCK_MIN:.0f} min ({LABELS[model_sel]})", f"{r[model_sel]:.2f} L/h",
                help="Rango probable P10–P90 calculado con los errores en validación: "
                     f"{max(r[model_sel]+RQ[model_sel][0],0):.2f}–{r[model_sel]+RQ[model_sel][1]:.2f} L/h")
    k[1].metric("Litros estimados hasta este bloque", f"{so_far[f'L_{model_sel}'].sum():.3f} L")
    k[2].metric("Costo estimado hasta este bloque", money(so_far[f'L_{model_sel}'].sum() * price))
    k[3].metric("Real (se conoce al final del bloque)", f"{so_far['L_real'].sum():.3f} L")

    st.subheader("Total del trayecto: integrando los pronósticos")
    rows = []
    for m in ["real"] + show:
        L = blk[f"L_{m}"].sum()
        rows.append({"Método": LABELS[m], "Litros": f"{L:.3f}", f"Costo ({currency})": f"{L * price:.2f}",
                     "Error vs real (%)": "—" if m == "real" else f"{100 * (L / blk['L_real'].sum() - 1):+.1f}"})
    c1, c2 = st.columns([1, 1.3])
    c1.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    c2.plotly_chart(cumulative_figure(blk, show), width="stretch")

# --------------------------------------------------------------------------
# 2. Asistente con LLM
# --------------------------------------------------------------------------
@st.cache_data
def demo_cache():
    p = C.ART_DIR / "llm" / "demo_cache.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def answer_question(q, state):
    """Respuesta en vivo si hay llave. Si no la hay o la API falla, usa la respuesta guardada."""
    cached = demo_cache().get(f"{state.veh}|{state.trip}|{state.block}|{q}")
    if assistant.has_api_key():
        import anthropic
        try:
            return {**assistant.ask(q, state), "fuente": "en vivo"}
        except anthropic.AuthenticationError:
            return {"texto": "La llave de API no es válida. Revisa ANTHROPIC_API_KEY en el archivo .env.",
                    "herramientas": [], "fuente": "error"}
        except anthropic.APIError as e:
            if cached:
                return {**cached, "fuente": "guardada, la API no respondió"}
            return {"texto": f"El asistente no respondió ({type(e).__name__}). Intenta de nuevo en unos segundos.",
                    "herramientas": [], "fuente": "error"}
    if cached:
        return {**cached, "fuente": "guardada"}
    return {"texto": "Sin conexión al LLM. En este modo solo hay respuestas guardadas para las preguntas "
                     "sugeridas en el trayecto que la app abre por defecto.", "herramientas": [], "fuente": "sin conexión"}


with tabs[1]:
    st.header("Asistente de consumo")
    st.write("Pregunta en español sobre el trayecto abierto en el simulador. Claude Haiku 5.5 interpreta la "
             "pregunta y consulta el modelo GRU con herramientas. Todas las cifras salen de esas herramientas.")
    state = assistant.TripState(int(veh), int(trip), int(now_block))
    live = assistant.has_api_key()
    st.caption(f"Contexto: vehículo {veh}, minuto {state.minute:.0f}. Cambia el trayecto o el minuto en el simulador. "
               + ("Modo en vivo." if live else "Modo sin conexión: se muestran respuestas guardadas.")
               + f" El costo usa el precio por defecto de {C.FUEL_PRICE_USD_L:.3f} USD/L.")
    chat = st.session_state.setdefault("chat", [])
    picked = None
    for c, q in zip(st.columns(len(assistant.SUGGESTED)), assistant.SUGGESTED):
        if c.button(q, width="stretch"):
            picked = q
    history = st.container()
    typed = st.chat_input("Escribe tu pregunta")
    question = typed or picked
    if question:
        with st.spinner("Consultando el modelo..."):
            chat.append({"pregunta": question, "minuto": state.minute, **answer_question(question, state)})
    with history:
        for item in chat:
            st.chat_message("user").write(item["pregunta"])
            with st.chat_message("assistant"):
                st.write(item["texto"])
                if item["herramientas"]:
                    with st.expander("Cómo se obtuvo la respuesta"):
                        info = f"Fuente: {item['fuente']}. Minuto {item['minuto']:.0f} del trayecto."
                        if item.get("latencia_s"):
                            info += (f" Latencia {item['latencia_s']:.1f} s, "
                                     f"{item['tokens_entrada'] + item['tokens_salida']:,} tokens, "
                                     f"{item['costo_usd']:.5f} USD.")
                        st.caption(info)
                        for call in item["herramientas"]:
                            st.markdown(f"**{call['name']}** `{json.dumps(call['input'], ensure_ascii=False)}`")
                            st.json(call["output"], expanded=False)
    if chat and st.button("Borrar conversación"):
        st.session_state["chat"] = []
        st.rerun()

    llm_metrics = load_json("llm/metrics.json")
    if llm_metrics:
        with st.expander("Evaluación del asistente: 40 preguntas, 20 directas y 20 variadas"):
            st.dataframe(pd.DataFrame([{
                "Configuración": m["cambio"], "Modelo": m["modelo"], "Exactitud (%)": m["exactitud_pct"],
                "Directas (%)": m["exactitud_directas_pct"], "Variadas (%)": m["exactitud_variadas_pct"],
                "Con cifras no trazables (%)": m["respuestas_con_cifras_no_trazables_pct"],
                "Latencia media (s)": m["latencia_media_s"], "Costo por consulta (USD)": m["costo_por_consulta_usd"]}
                for m in llm_metrics.values()]).style.format(precision=3), hide_index=True, width="stretch")
            st.caption("Detalle por caso en artifacts/llm/. Se reproduce con python -m src.llm_eval.")

# --------------------------------------------------------------------------
# 3. Real vs predicho
# --------------------------------------------------------------------------
with tabs[2]:
    st.header("Comparación entre consumo real y predicho (conjunto de prueba)")
    n = metrics["n_test"]
    st.caption(f"{n['vehicles']} vehículos no vistos · {n['trips']:,} trayectos · {n['blocks']:,} "
               f"pronósticos de {BLOCK_MIN:.0f} min · {n['liters']:.0f} L reales.")
    level = st.radio("Nivel", ["Litros por trayecto", f"Consumo por bloque de {BLOCK_MIN:.0f} min (L/h)"],
                     horizontal=True)
    comp = [model_sel, baseline_sel]
    cols = st.columns(2)
    for c, m in zip(cols, comp):
        if level.startswith("Litros"):
            x, y, unit = trips_test["L_real"], trips_test[f"L_{m}"], "L"
        else:
            x, y, unit = blocks_test["real"], blocks_test[m], "L/h"
        lim = float(np.quantile(x, 0.995))
        fig = go.Figure(go.Scattergl(x=x, y=y, mode="markers", marker=dict(size=4, color=COLORS[m], opacity=0.35),
                                     name=LABELS[m]))
        fig.add_trace(go.Scatter(x=[0, lim], y=[0, lim], mode="lines", line=dict(color="#111", dash="dash"),
                                 name="Predicción perfecta"))
        fig.update_layout(title=LABELS[m], height=380, xaxis_title=f"Real ({unit})",
                          yaxis_title=f"Predicho ({unit})", xaxis_range=[0, lim], yaxis_range=[0, lim],
                          margin=dict(l=10, r=10, t=40, b=10), showlegend=False)
        c.plotly_chart(fig, width="stretch")
    st.subheader("Distribución del error relativo por trayecto")
    fig = go.Figure()
    for m in METHODS:
        e = 100 * (trips_test[f"L_{m}"] / trips_test["L_real"] - 1)
        fig.add_trace(go.Histogram(x=e.clip(-60, 60), name=LABELS[m], opacity=0.55, nbinsx=60,
                                   marker_color=COLORS[m]))
    fig.update_layout(barmode="overlay", height=320, xaxis_title="Error (%) = predicho/real − 1",
                      margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, width="stretch")

# --------------------------------------------------------------------------
# 4. Rendimiento de modelos
# --------------------------------------------------------------------------
with tabs[3]:
    st.header("Rendimiento de los modelos LSTM y GRU")
    split = st.radio("Conjunto", ["test", "val"], horizontal=True,
                     format_func=lambda s: {"test": "Prueba", "val": "Validación"}[s])
    rows = []
    for m in METHODS:
        b, t = metrics["block"][split][m], metrics["trip"][split][m]
        row = {"Método": LABELS[m], "MAE (L/h)": b["MAE"], "RMSE (L/h)": b["RMSE"], "Sesgo (L/h)": b["bias"],
               "MAE trayecto (L)": t["MAE_L"], "Error mediano trayecto (%)": t["MAPE_pct"],
               "Error total flota (%)": t["total_error_pct"]}
        if m in metrics.get("inference", {}):
            mi, inf = train_sum["models"][m], metrics["inference"][m]
            row.update({"Parámetros": mi["params"], "MFLOPs/pronóstico": mi["flops_per_forecast"] / 1e6,
                        "Latencia CPU (ms)": inf["latency_ms_cpu_batch1"],
                        "Entrenamiento (s)": mi["train_time_s"], "Épocas": mi["epochs_run"]})
        rows.append(row)
    df = pd.DataFrame(rows)
    st.dataframe(df.style.format(precision=3, na_rep="—").format(
        {"Parámetros": "{:,.0f}", "Épocas": "{:.0f}", "Entrenamiento (s)": "{:.0f}"}, na_rep="—").highlight_min(
        subset=["MAE (L/h)", "RMSE (L/h)", "MAE trayecto (L)"], color="#bbf7d0"),
        hide_index=True, width="stretch")
    st.caption("MAE/RMSE: error del pronóstico del consumo medio de los próximos "
               f"{BLOCK_MIN:.0f} min. Error por trayecto: litros integrados vs reales. "
               f"Entrenado en: {train_sum['models'][model_sel]['device']}.")
    c1, c2 = st.columns(2)
    fig = go.Figure()
    for m in ["lstm", "gru"]:
        h = pd.DataFrame(train_sum["models"][m]["history"])
        fig.add_trace(go.Scatter(x=h["epoch"], y=h["val_mae"], name=f"{LABELS[m]} val MAE",
                                 line=dict(color=COLORS[m])))
    fig.add_hline(y=metrics["block"]["val"]["persist"]["MAE"], line=dict(dash="dot", color=COLORS["persist"]),
                  annotation_text="Persistencia")
    fig.add_hline(y=metrics["block"]["val"]["hist"]["MAE"], line=dict(dash="dot", color=COLORS["hist"]),
                  annotation_text="Promedio histórico")
    fig.update_layout(title="Curvas de aprendizaje (validación)", height=330, xaxis_title="Época",
                      yaxis_title="MAE (L/h)", margin=dict(l=10, r=10, t=40, b=10))
    c1.plotly_chart(fig, width="stretch")
    bt = blocks_test.copy()
    bt["min_bin"] = pd.cut(bt["t"] * STEP_MIN, [-1, 2, 6, 12, 20, 30, 1e9],
                           labels=["0–2", "2–6", "6–12", "12–20", "20–30", ">30"])
    err = bt.groupby("min_bin", observed=True)[METHODS + ["real"]].apply(
        lambda g: pd.Series({LABELS[m]: np.mean(np.abs(g[m] - g["real"])) for m in METHODS}))
    fig = go.Figure([go.Bar(x=err.index.astype(str), y=err[LABELS[m]], name=LABELS[m], marker_color=COLORS[m])
                     for m in METHODS])
    fig.update_layout(title="MAE según minuto del trayecto", height=330, xaxis_title="Minuto en que se emite",
                      yaxis_title="MAE (L/h)", margin=dict(l=10, r=10, t=40, b=10))
    c2.plotly_chart(fig, width="stretch")
    it = C.ART_DIR / "iterations.csv"
    if it.exists():
        st.subheader("Historial de iteraciones")
        st.dataframe(pd.read_csv(it).fillna(""), hide_index=True, width="stretch")

# --------------------------------------------------------------------------
# 5. Impacto económico
# --------------------------------------------------------------------------
with tabs[4]:
    st.header("Impacto económico de mejorar las predicciones")
    st.markdown(
        "**Decisión:** reservar `R = P · (1 + m)` litros para cada trayecto, donde `P` es el consumo "
        "predicho y `m` el margen de reserva.  \n"
        "**Valor por trayecto** = ahorro por acierto (si R ≥ real) − costo de déficit (si R < real) − "
        "costo de exceso (litros sobrantes).  \n"
        "El margen óptimo de cada método se elige en **validación** y el resultado se mide en **prueba**.")
    Av, At = trips_val["L_real"].to_numpy(), trips_test["L_real"].to_numpy()
    res = {}
    for m in METHODS:
        cv = value_curve(Av, trips_val[f"L_{m}"].to_numpy(), PRM)
        m_star = float(cv.loc[cv["value_per_trip"].idxmax(), "margin"])
        res[m] = outcome(At, trips_test[f"L_{m}"].to_numpy(), m_star, PRM)
        res[m]["sigma"] = log_error_sigma(At, trips_test[f"L_{m}"].to_numpy())
    base = res[baseline_sel]
    k = st.columns(4)
    gain = 1000 * (res[model_sel]["value_per_trip"] - base["value_per_trip"])
    k[0].metric(f"Valor {LABELS[model_sel]} vs {LABELS[baseline_sel]}", money(gain), help="Por cada 1 000 trayectos")
    k[1].metric("Margen óptimo (validación)", f"{100*res[model_sel]['margin']:+.0f} %",
                f"{100*(res[model_sel]['margin']-base['margin']):+.0f} pp vs referencia", delta_color="inverse")
    k[2].metric("Emergencias en prueba", res[model_sel]["emergencies"],
                f"{res[model_sel]['emergencies']-base['emergencies']:+d} vs referencia", delta_color="inverse")
    exc_diff = 1000 * (res[model_sel]["excess_L_per_trip"] - base["excess_L_per_trip"])
    k[3].metric("Litros sobrantes / 1 000 trayectos", f"{1000*res[model_sel]['excess_L_per_trip']:.0f} L",
                f"{exc_diff:+.0f} L vs referencia", delta_color="inverse")

    rows = [{"Método": LABELS[m], "Margen elegido": f"{100*r['margin']:+.0f} %",
             "Cobertura (%)": r["coverage_pct"], "Emergencias": r["emergencies"],
             "Reserva media (L)": r["reserved_L_per_trip"], "Sobrante medio (L)": r["excess_L_per_trip"],
             f"Valor/trayecto ({currency})": r["value_per_trip"],
             f"Δ vs {LABELS[baseline_sel]} por 1 000 trayectos": 1000 * (r["value_per_trip"] - base["value_per_trip"]),
             "σ error log": r["sigma"]} for m, r in res.items()]
    st.dataframe(pd.DataFrame(rows).style.format(precision=3), hide_index=True, width="stretch")

    c1, c2 = st.columns(2)
    fig = go.Figure()
    for m in METHODS:
        cv = value_curve(At, trips_test[f"L_{m}"].to_numpy(), PRM)
        fig.add_trace(go.Scatter(x=100 * cv["margin"], y=cv["value_per_trip"], name=LABELS[m],
                                 line=dict(color=COLORS[m])))
        fig.add_trace(go.Scatter(x=[100 * res[m]["margin"]], y=[res[m]["value_per_trip"]], mode="markers",
                                 marker=dict(color=COLORS[m], size=10, symbol="star"), showlegend=False))
    fig.update_layout(title="Valor esperado por trayecto según margen de reserva (prueba)", height=380,
                      xaxis_title="Margen de reserva (%)", yaxis_title=f"Valor por trayecto ({currency})",
                      margin=dict(l=10, r=10, t=40, b=10))
    c1.plotly_chart(fig, width="stretch")

    ec = error_value_curve(At, PRM)
    fig = go.Figure(go.Scatter(x=100 * ec["sigma"], y=ec["value_per_trip"], name="Curva teórica",
                               line=dict(color="#111")))
    for m in METHODS:
        y = float(np.interp(res[m]["sigma"], ec["sigma"], ec["value_per_trip"]))
        fig.add_trace(go.Scatter(x=[100 * res[m]["sigma"]], y=[y], mode="markers", name=LABELS[m],
                                 marker=dict(size=11, color=COLORS[m])))
    fig.update_layout(title="Curva de valor por nivel de error de predicción", height=380,
                      xaxis_title="Error de predicción (σ del error logarítmico, %)",
                      yaxis_title=f"Valor máximo por trayecto ({currency})", margin=dict(l=10, r=10, t=40, b=10))
    c2.plotly_chart(fig, width="stretch")

    st.subheader("Margen necesario para una cobertura objetivo")
    target = st.slider("Cobertura objetivo (% de trayectos sin quedarse corto)", 80.0, 99.9, 99.0, 0.1)
    rows = []
    for m in METHODS:
        ratio_v = trips_val["L_real"] / trips_val[f"L_{m}"].clip(lower=1e-6)
        mreq = float(np.quantile(ratio_v, target / 100) - 1)          # elegido en validación
        R = trips_test[f"L_{m}"] * (1 + mreq)
        rows.append({"Método": LABELS[m], "Margen requerido (val)": f"{100*mreq:+.1f} %",
                     "Cobertura lograda en prueba (%)": 100 * float((R >= trips_test["L_real"]).mean()),
                     "Sobrante por 1 000 trayectos (L)": 1000 * float((R - trips_test["L_real"]).clip(lower=0).mean()),
                     f"Capital inmovilizado / 1 000 trayectos ({currency})":
                         1000 * price * float((R - trips_test["L_real"]).clip(lower=0).mean())})
    st.dataframe(pd.DataFrame(rows).style.format(precision=2), hide_index=True, width="stretch")
    st.info("Los parámetros económicos son supuestos documentados (ver barra lateral y pestaña de "
            "metodología). La app no afirma ahorros verificados: muestra cómo cambia el valor esperado "
            "según el error de predicción bajo esos supuestos.")

# --------------------------------------------------------------------------
# 6. CSV propio
# --------------------------------------------------------------------------
def load_upload(up, disp):
    """Lee y valida el CSV subido. Devuelve (pasos de 10 s, mensaje de error)."""
    from src.preprocess import RAW_COLS, process_records_to_bins
    raw = pd.read_csv(up).rename(columns=RAW_COLS)
    missing = {"ts_ms", "speed", "rpm", "load"} - set(raw.columns)
    if missing:
        return None, f"Faltan columnas: {sorted(missing)}"
    num = [c for c in RAW_COLS.values() if c in raw]
    raw[num] = raw[num].apply(pd.to_numeric, errors="coerce")
    raw = raw.dropna(subset=["ts_ms"])
    if raw.empty:
        return None, "La columna Timestamp(ms) no tiene valores numéricos."
    raw["ts_ms"] = raw["ts_ms"] - raw["ts_ms"].min()
    ub = process_records_to_bins(raw, disp)
    if len(ub) < 2 or ub["fcr"].isna().all():
        return None, "El trayecto es demasiado corto o no tiene datos de consumo válidos."
    empty = [c for c in ["speed", "rpm", "load"] if ub[c].isna().all()]
    if empty:
        return None, f"Estas columnas no tienen valores numéricos: {empty}"
    return ub, None


with tabs[5]:
    st.header("Probar con tu propio trayecto")
    st.write("Sube un CSV de **un trayecto** con el formato de VED (registros OBD-II). Columnas mínimas: "
             "`Timestamp(ms)`, `Vehicle Speed[km/h]`, `Engine RPM[RPM]`, `Absolute Load[%]` y "
             "`MAF[g/sec]` o `Fuel Rate[L/hr]`. Fuel trims son opcionales.")
    sample = ROOT / "app" / "sample_trip.csv"
    if sample.exists():
        st.download_button("Descargar CSV de ejemplo", sample.read_bytes(), "sample_trip.csv", "text/csv")
    up = st.file_uploader("CSV del trayecto", type="csv")
    c1, c2 = st.columns(2)
    disp = c1.number_input("Cilindrada del motor en litros, solo si no hay MAF", 0.6, 8.0, 2.0, 0.1)
    hist_rate = c2.number_input("Consumo promedio histórico de tu vehículo (L/h)", 0.1, 50.0,
                                round(metrics["train_mean_rate_Lh"], 2), 0.1)
    if up is not None:
        from src.inference import forecast_trip
        try:
            ub, msg = load_upload(up, disp)
        except Exception as e:  # noqa: BLE001
            ub, msg = None, f"No se pudo leer el archivo: {e}"
        if msg:
            st.error(msg)
        else:
            models, scaler, residual = bundle()
            blk_u = forecast_trip(ub, models, scaler, residual, hist_rate)
            nb = st.slider("Bloque actual", 0, len(blk_u) - 1, len(blk_u) - 1, key="ub") if len(blk_u) > 1 else 0
            st.plotly_chart(trip_figure(ub, blk_u, nb, [model_sel, baseline_sel]), width="stretch")
            rows = [{"Método": LABELS[m], "Litros": blk_u[f"L_{m}"].sum(),
                     f"Costo ({currency})": blk_u[f"L_{m}"].sum() * price} for m in ["real", model_sel, baseline_sel]]
            st.dataframe(pd.DataFrame(rows).style.format(precision=3), hide_index=True)
            st.caption("Real es el consumo estimado con el Algoritmo 1 de VED a partir de MAF y fuel trims.")

# --------------------------------------------------------------------------
# 7. Datos y metodología
# --------------------------------------------------------------------------
with tabs[6]:
    st.header("Datos, pipeline y supuestos")
    if quality:
        k = st.columns(5)
        k[0].metric("Vehículos de gasolina usados", quality["vehicles_kept"], f"de {quality['ice_vehicles_static']}")
        k[1].metric("Trayectos válidos", f"{quality['trips_kept']:,}", f"de {quality['trips_total']:,}")
        k[2].metric("Horas de conducción", f"{quality['hours_kept']:,.0f}")
        k[3].metric("Litros consumidos", f"{quality['liters_kept']:,.0f}")
        k[4].metric("Registros OBD-II (ICE)", f"{quality['ice_raw_records']/1e6:.1f} M")
        c1, c2 = st.columns(2)
        cov = pd.Series(quality["raw_coverage_frac"]).rename("Cobertura") * 100
        c1.write("**Cobertura de variables crudas (% de registros ICE)**")
        c1.dataframe(cov.round(2).to_frame(), width="stretch")
        src = pd.Series(quality["fcr_source_records"]).rename("Registros")
        c2.write("**Fuente de la tasa de consumo (Algoritmo 1 de VED)**")
        c2.dataframe(src.to_frame(), width="stretch")
        st.caption(f"División por vehículo: {quality['split_vehicles']} · trayectos: {quality['split_trips']}")
    st.graphviz_chart("""
    digraph G { rankdir=LR; node [shape=box, style="rounded,filled", fillcolor="#eef2ff", fontname=Helvetica];
      a [label="VED (54 CSV semanales)\\n+ datos estáticos"]; b [label="Filtrar gasolina (ICE)"];
      c [label="Algoritmo 1 VED\\nFuel Rate / MAF·trims / Carga·RPM"]; d [label="Agregar a pasos de 10 s\\ninterpolar huecos ≤30 s"];
      e [label="Filtro de calidad\\n≥90 % válido, ≥3 min"]; f [label="Split por vehículo\\n70/15/15"];
      g [label="Escalado (solo train)\\nVentanas 3 min → 2 min"]; h [label="LSTM / GRU\\n(+ baselines)"];
      i [label="Integrar pronósticos\\nlitros y costo"]; j [label="Modelo de valor\\nmargen de reserva"]; k [label="App Streamlit"];
      l [label="Asistente Claude Haiku 5.5\\nherramientas sobre GRU"];
      a->b->c->d->e->f->g->h->i->j->k; h->l->k; }""")
    st.markdown(f"""
**Supuestos y fuentes**
- Consumo: VED no reporta `Fuel Rate` para casi ningún vehículo de gasolina; se estima con el Algoritmo 1 de
  Oh et al. (2022): `combustible[g/s] = MAF·(1+STFT/100+LTFT/100)/AFR`, AFR = {C.AFR_E10} (E10). Conversión a L/h
  con densidad {C.FUEL_DENSITY_G_L:.0f} g/L (supuesto).
- Precio por defecto: EIA, *Gasoline and Diesel Fuel Update*, gasolina regular Midwest (PADD 2), 26-oct-2026:
  4.269 USD/gal = {C.FUEL_PRICE_USD_L:.3f} USD/L.
- Ahorro por acierto: cargo de asistencia por falta de combustible para no miembros 50–150 USD (Engineer Fix,
  dic-2025) → 100 USD. Costo fijo de déficit: recargos nocturnos/fin de semana 20–50 USD → 35 USD. El combustible
  entregado se cobra a precio de mercado → prima 0.
- Costo de exceso ({100*C.EXCESS_COST_FRAC:.0f} % del precio por litro sobrante): supuesto del equipo.
- Pronóstico: historia {C.LOOKBACK*C.STEP_S//60} min, horizonte {BLOCK_MIN:.0f} min, actualización cada {BLOCK_MIN:.0f} min.
  No se usan velocidad, RPM ni carga futuras.

**Cuándo no usar esta herramienta:** vehículos híbridos/eléctricos o diésel, trayectos de carretera muy largos
con perfiles distintos a Ann Arbor, o decisiones de seguridad (el rango P10–P90 indica la incertidumbre típica).
""")
