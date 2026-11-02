"""FuelWise: aplicación web en Streamlit.

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

st.set_page_config(page_title="FuelWise", page_icon=":material/local_gas_station:", layout="wide")

# Paleta con roles fijos: el modelo principal en verde azulado, la referencia actual en ámbar,
# lo medido en gris oscuro y lo secundario en grises.
INK, MUTED, GRID = "#1F2A33", "#5F6B75", "#E4E8EB"
COLORS = {"real": "#3A4650", "gru": "#1B7F79", "lstm": "#8CC3BE", "hist": "#D9922E", "persist": "#A3ADB5"}
LABELS = {"gru": "GRU", "lstm": "LSTM", "hist": "Promedio histórico", "persist": "Persistencia", "real": "Medido"}
STEP_MIN = C.STEP_S / 60.0
BLOCK_MIN = C.HORIZON * C.STEP_S / 60.0

st.html("""
<style>
.block-container {padding-top: 1.6rem; max-width: 1320px;}
header[data-testid="stHeader"] {background: transparent; height: 0;}
.num {font-family: 'IBM Plex Mono', Consolas, monospace; font-variant-numeric: tabular-nums;}
.brand {display: flex; align-items: baseline; gap: 14px; margin-bottom: 2px;}
.brand b {font-size: 1.35rem; letter-spacing: -0.01em; color: #0F2A3D;}
.brand span {color: #5F6B75; font-size: 0.95rem;}
.readout {border-top: 2px solid #0F2A3D; padding-top: 10px;}
.readout .lbl {color: #5F6B75; font-size: 0.9rem;}
.readout .big {font-size: 2.6rem; line-height: 1.1; color: #0F2A3D; margin: 2px 0;}
.readout .big small {font-size: 1rem; color: #5F6B75; margin-left: 4px;}
.readout .rng {font-size: 0.92rem; color: #1F2A33; margin-bottom: 12px;}
.readout dl {margin: 0; padding: 0; border-top: 1px solid #E4E8EB;}
.readout dt, .readout dd {margin: 0; padding: 0;}
.readout dl div {display: flex; justify-content: space-between; padding: 6px 0; border-bottom: 1px solid #E4E8EB;}
.readout dt {color: #5F6B75; font-size: 0.9rem;}
.readout dd {margin: 0; font-size: 0.95rem;}
.note {color: #5F6B75; font-size: 0.88rem;}
.lead {font-size: 1.05rem; max-width: 70ch; margin-bottom: 0.6rem;}
.st-key-sugeridas button {min-height: 0; padding: 3px 0; text-align: left; justify-content: flex-start;}
.st-key-sugeridas button p {font-size: 0.92rem;}
.st-key-sugeridas {gap: 0 !important;}
</style>
""")


# --------------------------------------------------------------------------
# Carga de artefactos
# --------------------------------------------------------------------------
@st.cache_data
def load_json(name):
    p = C.ART_DIR / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


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
    st.error("No hay modelos entrenados en artifacts/. Ejecuta python run_all.py.")
    st.stop()

train_sum = load_json("training_summary.json")
quality = load_json("quality_report.json")
llm_metrics = load_json("llm/metrics.json")
blocks_test, blocks_val = load_pq("blocks_test.parquet"), load_pq("blocks_val.parquet")
trips_test, trips_val = load_pq("trips_test.parquet"), load_pq("trips_val.parquet")
test_bins = load_pq("test_bins.parquet")
METHODS = [m for m in ["gru", "lstm", "hist", "persist"] if m in blocks_test]


@st.cache_data
def residual_quantiles():
    """Rango P10 a P90 del error por bloque en validación."""
    return {m: (float(np.quantile(blocks_val["real"] - blocks_val[m], 0.10)),
                float(np.quantile(blocks_val["real"] - blocks_val[m], 0.90))) for m in METHODS}


RQ = residual_quantiles()

# Supuestos económicos: se editan en la pestaña Reserva y se leen en todas.
DEFAULTS = {"price": float(C.FUEL_PRICE_USD_L), "saving_hit": C.SAVING_PER_HIT_USD, "deficit_fixed": C.DEFICIT_FIXED_USD,
            "deficit_premium": C.DEFICIT_PREMIUM_FRAC, "excess_frac": C.EXCESS_COST_FRAC, "currency": "USD"}
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)
price, currency = st.session_state["price"], st.session_state["currency"]


def thousands(x):
    """Entero con espacio no separable como separador de miles."""
    return f"{x:,.0f}".replace(",", "\u00a0")


def money(x):
    return f"{x:,.2f} {currency}"


def style_fig(fig, height, ytitle=None, xtitle=None, legend=True):
    fig.update_layout(height=height, margin=dict(l=8, r=8, t=8, b=8), plot_bgcolor="white", paper_bgcolor="white",
                      font=dict(family="IBM Plex Sans, Segoe UI, sans-serif", size=13, color=INK),
                      showlegend=legend, legend=dict(orientation="h", y=1.02, yanchor="bottom", x=0, title=None,
                                                     font=dict(size=12, color=MUTED)),
                      hoverlabel=dict(font_family="IBM Plex Mono, Consolas, monospace"))
    fig.update_xaxes(title=xtitle, showgrid=False, linecolor="#C9D0D5", ticks="outside", tickcolor="#C9D0D5",
                     title_font=dict(size=12, color=MUTED))
    fig.update_yaxes(title=ytitle, gridcolor=GRID, zeroline=False, title_font=dict(size=12, color=MUTED))
    return fig


st.html('<div class="brand"><b>FuelWise</b><span>Pronóstico de consumo de combustible por trayecto, '
        'con una red GRU y un asistente que responde preguntas.</span></div>')

tabs = st.tabs(["Trayecto", "Modelo", "Asistente", "Reserva", "Datos y método", "Subir trayecto"])


# --------------------------------------------------------------------------
# Gráfica de un trayecto (la usan el trayecto de prueba y el CSV subido)
# --------------------------------------------------------------------------
def trip_figure(bins, blk, now_block, show, height=400):
    fig = go.Figure()
    tmin = bins["step"].to_numpy() * STEP_MIN
    now_min = now_block * BLOCK_MIN
    seen = tmin < now_min
    fcr = bins["fcr"].to_numpy()
    fig.add_trace(go.Scatter(x=tmin[seen], y=fcr[seen], name="Medido", mode="lines",
                             line=dict(color=COLORS["real"], width=1.4)))
    fig.add_trace(go.Scatter(x=tmin[~seen], y=fcr[~seen], name="Aún no ocurre", hoverinfo="skip", mode="lines",
                             line=dict(color="#C9D0D5", width=1, dash="dot")))
    cur = blk[blk["t"] == now_block * C.HORIZON]
    if len(cur):
        r = cur.iloc[0]
        lo, hi = RQ[show[0]]
        x0, x1 = r["t"] * STEP_MIN, (r["t"] + r["block_len"]) * STEP_MIN
        fig.add_trace(go.Scatter(x=[x0, x1, x1, x0], y=[max(r[show[0]] + lo, 0)] * 2 + [r[show[0]] + hi] * 2,
                                 fill="toself", fillcolor="rgba(27,127,121,0.12)", line=dict(width=0), mode="lines",
                                 name="Rango probable", hoverinfo="skip"))
    for m in show:
        xs, ys = [], []
        for _, r in blk[blk["t"] <= now_block * C.HORIZON].iterrows():
            x0 = r["t"] * STEP_MIN
            xs += [x0, x0 + r["block_len"] * STEP_MIN, None]
            ys += [r[m], r[m], None]
        fig.add_trace(go.Scatter(x=xs, y=ys, name=LABELS[m], mode="lines",
                                 line=dict(color=COLORS[m], width=3 if m == show[0] else 2,
                                           dash="solid" if m in ("gru", "lstm") else "dash")))
    fig.add_vline(x=now_min, line=dict(color="#0F2A3D", width=1))
    fig.add_annotation(x=now_min, y=1, yref="paper", text=f"minuto {now_min:.0f}", showarrow=False,
                       xanchor="left", xshift=4, font=dict(size=12, color="#0F2A3D"))
    return style_fig(fig, height, "L/h", "Minuto del trayecto")


def readout(forecast, lo, hi, rows):
    items = "".join(f"<div><dt>{k}</dt><dd class='num'>{v}</dd></div>" for k, v in rows)
    return (f"<div class='readout'><div class='lbl'>Consumo esperado en los próximos {BLOCK_MIN:.0f} minutos</div>"
            f"<div class='big num'>{forecast:.2f}<small>L/h</small></div>"
            f"<div class='rng'>Rango probable de <span class='num'>{lo:.1f}</span> a <span class='num'>{hi:.1f}</span> L/h</div>"
            f"<dl>{items}</dl></div>")


# --------------------------------------------------------------------------
# Asistente
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
    return {"texto": "Sin conexión al asistente. Sin internet solo hay respuestas guardadas para las preguntas "
                     "sugeridas en el trayecto que la app abre por defecto.", "herramientas": [], "fuente": "sin conexión"}


def render_answer(item):
    st.markdown(f"**{item['pregunta']}**")
    st.write(item["texto"])
    if item["herramientas"]:
        with st.expander("De dónde salen las cifras"):
            info = f"Minuto {item['minuto']:.0f}. Respuesta {item['fuente']}."
            if item.get("latencia_s"):
                info += (f" {item['latencia_s']:.1f} s, {item['tokens_entrada'] + item['tokens_salida']:,} tokens, "
                         f"{item['costo_usd']:.5f} USD.")
            st.caption(info)
            for call in item["herramientas"]:
                st.markdown(f"`{call['name']}` con `{json.dumps(call['input'], ensure_ascii=False)}`")
                st.json(call["output"], expanded=False)


# --------------------------------------------------------------------------
# 1. Trayecto: simulador y asistente en la misma vista
# --------------------------------------------------------------------------
with tabs[0]:
    tt = trips_test
    demo_veh, demo_trip, _ = assistant.demo_trip()
    veh_opts = sorted(tt["veh"].unique())
    c1, c2, c3 = st.columns([1.1, 1.3, 2])
    veh = c1.selectbox("Vehículo", veh_opts, index=veh_opts.index(demo_veh),
                       format_func=lambda v: f"{v}, motor {tt.loc[tt.veh == v, 'engine'].iloc[0]}")
    tv = tt[tt["veh"] == veh].sort_values("duration_min", ascending=False)
    trip = c2.selectbox("Trayecto", tv["trip"].tolist(),
                        index=tv["trip"].tolist().index(demo_trip) if veh == demo_veh else 0,
                        format_func=lambda t: (lambda r: f"{t}, {r.duration_min:.0f} min, {r.distance_km:.1f} km")(
                            tv[tv.trip == t].iloc[0]))
    c3.markdown("<div class='note' style='padding-top:1.9rem'>Vehículos que el modelo no vio al entrenar. "
                "Cada 2 minutos pronostica con los últimos 3 minutos de velocidad, RPM, carga y consumo.</div>",
                unsafe_allow_html=True)

    bins = test_bins[(test_bins.veh == veh) & (test_bins.trip == trip)].sort_values("step")
    blk = blocks_test[(blocks_test.veh == veh) & (blocks_test.trip == trip)].sort_values("t").reset_index(drop=True)
    n_blocks = len(blk)

    left, right = st.columns([2.15, 1], gap="large")
    with left:
        last_min = int((n_blocks - 1) * BLOCK_MIN)
        minute = st.slider("Minuto del trayecto", 0, max(last_min, int(BLOCK_MIN)), min(6, last_min),
                           step=int(BLOCK_MIN), format="%d min") if n_blocks > 1 else 0
        now_block = int(minute // BLOCK_MIN)
        compare = st.toggle("Comparar con LSTM y persistencia", value=False)
        show = ["gru", "hist"] + (["lstm", "persist"] if compare else [])
        st.plotly_chart(trip_figure(bins, blk, now_block, show), width="stretch", config={"displayModeBar": False})

        lit_real = float(bins.loc[bins["step"] < now_block * C.HORIZON, "fcr"].sum() * STEP_MIN / 60)
        with st.expander("Total del trayecto, sumando los pronósticos"):
            rows = [{"Método": LABELS[m], "Litros": blk[f"L_{m}"].sum(), f"Costo ({currency})": blk[f"L_{m}"].sum() * price,
                     "Diferencia con lo medido": f"{100 * (blk[f'L_{m}'].sum() / blk['L_real'].sum() - 1):+.1f} %"
                     if m != "real" else ""} for m in ["real"] + show]
            st.dataframe(pd.DataFrame(rows).style.format({"Litros": "{:.3f}", f"Costo ({currency})": "{:.2f}"}),
                         hide_index=True, width="stretch")
            st.caption("La persistencia suele acercarse más en el total, porque repite el consumo medido con "
                       "3 minutos de retraso. El GRU es mejor en el pronóstico de los próximos 2 minutos.")

    with right:
        r = blk.iloc[now_block]
        lo, hi = max(r["gru"] + RQ["gru"][0], 0), r["gru"] + RQ["gru"][1]
        st.html(readout(r["gru"], lo, hi, [
            ("Medido hasta ahora", f"{lit_real:.2f} L"),
            ("Costo hasta ahora", money(lit_real * price)),
            ("Promedio histórico del vehículo", f"{r['hist']:.2f} L/h"),
            ("Medido en esos 2 minutos", f"{r['real']:.2f} L/h")]))

        st.markdown("#### Preguntar")
        state = assistant.TripState(int(veh), int(trip), int(now_block))
        chat = st.session_state.setdefault("chat", [])
        answers = st.container()
        picked = None
        with st.container(key="sugeridas"):
            for q in assistant.SUGGESTED:
                if st.button(q, type="tertiary"):
                    picked = q
        question = st.chat_input("Escribe una pregunta sobre este trayecto") or picked
        if question:
            with st.spinner("Consultando el modelo"):
                chat.append({"pregunta": question, "minuto": state.minute, **answer_question(question, state)})
        with answers:
            for item in chat[-3:]:
                render_answer(item)
        mode = "en vivo con Claude Haiku 5.5" if assistant.has_api_key() else "sin conexión, con respuestas guardadas"
        st.caption(f"Asistente {mode}. Las cifras salen de herramientas que consultan el GRU.")
        if chat and st.button("Borrar preguntas", type="tertiary"):
            st.session_state["chat"] = []
            st.rerun()


# --------------------------------------------------------------------------
# 2. Modelo
# --------------------------------------------------------------------------
with tabs[1]:
    bt = metrics["block"]["test"]
    gain = 100 * (1 - bt["gru"]["MAE"] / bt["hist"]["MAE"])
    n = metrics["n_test"]
    st.markdown(f"<p class='lead'>En {n['vehicles']} vehículos que no se usaron al entrenar, el GRU se equivoca en "
                f"promedio <span class='num'>{bt['gru']['MAE']:.2f}</span> L/h al pronosticar los próximos 2 minutos, "
                f"{gain:.0f} % menos que el promedio histórico. Prueba con {thousands(n['trips'])} trayectos y "
                f"{thousands(n['blocks'])} pronósticos.</p>", unsafe_allow_html=True)
    split = st.segmented_control("Conjunto", ["Prueba", "Validación"], default="Prueba")
    sk = "val" if split == "Validación" else "test"
    rows = []
    for m in METHODS:
        b, t = metrics["block"][sk][m], metrics["trip"][sk][m]
        row = {"Método": LABELS[m], "MAE L/h": b["MAE"], "RMSE L/h": b["RMSE"], "Sesgo L/h": b["bias"],
               "Error mediano por trayecto %": t["MAPE_pct"]}
        if m in metrics.get("inference", {}):
            mi, inf = train_sum["models"][m], metrics["inference"][m]
            row.update({"Parámetros": thousands(mi["params"]), "Latencia ms": f"{inf['latency_ms_cpu_batch1']:.2f}",
                        "Épocas": str(mi["epochs_run"])})
        rows.append(row)
    st.dataframe(pd.DataFrame(rows).fillna("").style.format(
        {"MAE L/h": "{:.3f}", "RMSE L/h": "{:.3f}", "Sesgo L/h": "{:.3f}", "Error mediano por trayecto %": "{:.1f}",
         }), hide_index=True, width="stretch")

    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("##### Medido contra pronosticado")
        comp = st.selectbox("Método", METHODS, format_func=LABELS.get, label_visibility="collapsed")
        x, y = blocks_test["real"], blocks_test[comp]
        lim = float(np.quantile(x, 0.995))
        fig = go.Figure(go.Scattergl(x=x, y=y, mode="markers", marker=dict(size=4, color=COLORS[comp], opacity=0.3),
                                     name=LABELS[comp]))
        fig.add_trace(go.Scatter(x=[0, lim], y=[0, lim], mode="lines", line=dict(color=INK, width=1, dash="dash"),
                                 name="Sin error"))
        fig.update_xaxes(range=[0, lim])
        fig.update_yaxes(range=[0, lim])
        st.plotly_chart(style_fig(fig, 360, "Pronosticado L/h", "Medido L/h", legend=False), width="stretch",
                        config={"displayModeBar": False})
    with c2:
        st.markdown("##### Aprendizaje en validación")
        fig = go.Figure()
        for m in ["gru", "lstm"]:
            h = pd.DataFrame(train_sum["models"][m]["history"])
            fig.add_trace(go.Scatter(x=h["epoch"], y=h["val_mae"], name=LABELS[m], mode="lines",
                                     line=dict(color=COLORS[m], width=2)))
        for m in ["hist", "persist"]:
            fig.add_hline(y=metrics["block"]["val"][m]["MAE"], line=dict(dash="dot", color=COLORS[m], width=1.5),
                          annotation_text=LABELS[m], annotation_font_color=MUTED, annotation_position="top left")
        st.plotly_chart(style_fig(fig, 400, "MAE L/h", "Época"), width="stretch", config={"displayModeBar": False})

    st.markdown("##### Error según el minuto del trayecto")
    b2 = blocks_test.copy()
    b2["min_bin"] = pd.cut(b2["t"] * STEP_MIN, [-1, 1, 6, 12, 20, 1e9], labels=["inicio", "2 a 6", "6 a 12", "12 a 20", "más de 20"])
    err = b2.groupby("min_bin", observed=True)[METHODS + ["real"]].apply(
        lambda g: pd.Series({m: float(np.mean(np.abs(g[m] - g["real"]))) for m in METHODS}))
    fig = go.Figure([go.Bar(x=err.index.astype(str), y=err[m], name=LABELS[m], marker_color=COLORS[m]) for m in METHODS])
    st.plotly_chart(style_fig(fig, 300, "MAE L/h", "Minuto en que se emite el pronóstico"), width="stretch",
                    config={"displayModeBar": False})
    st.caption("Al inicio no hay historia y el promedio histórico es mejor. Desde el minuto 2 el GRU lo supera.")

    it = C.ART_DIR / "iterations.csv"
    if it.exists():
        st.markdown("##### Iteraciones")
        st.dataframe(pd.read_csv(it).fillna(""), hide_index=True, width="stretch")


# --------------------------------------------------------------------------
# 3. Asistente: cómo se evaluó
# --------------------------------------------------------------------------
with tabs[2]:
    if not llm_metrics:
        st.info("Todavía no hay evaluación del asistente. Ejecuta python -m src.llm_eval.")
    else:
        h = llm_metrics["haiku_v2"]
        st.markdown(f"<p class='lead'>El asistente acierta <span class='num'>{h['exactitud_pct']:.1f} %</span> de 40 "
                    f"preguntas y no dio cifras sin respaldo. Responde en {h['latencia_media_s']:.1f} s por "
                    f"{h['costo_por_consulta_usd']:.4f} USD. Claude Haiku 5.5 interpreta la pregunta y las cifras "
                    "salen de cuatro herramientas que consultan el GRU.</p>", unsafe_allow_html=True)
        names = {"reglas": "Reglas sin LLM", "haiku_sin_herramientas": "Haiku sin herramientas",
                 "haiku_v1": "Haiku, prompt mínimo", "haiku_v2": "Haiku, final", "sonnet_v2": "Sonnet 5.5"}
        c1, c2 = st.columns([1.2, 1], gap="large")
        with c1:
            fig = go.Figure()
            keys = [k for k in names if k in llm_metrics]
            for grp, col, field in [("Directas", "#A3ADB5", "exactitud_directas_pct"),
                                    ("Variadas", "#1B7F79", "exactitud_variadas_pct")]:
                vals = [llm_metrics[k][field] for k in keys]
                fig.add_trace(go.Bar(x=[names[k].replace(" ", "<br>", 1) for k in keys], y=vals, name=grp,
                                     marker_color=col,
                                     text=[f"{v:.0f}" for v in vals], textposition="outside"))
            fig.update_yaxes(range=[0, 115])
            st.plotly_chart(style_fig(fig, 330, "Exactitud %"), width="stretch", config={"displayModeBar": False})
        with c2:
            st.markdown("Las 20 preguntas **directas** usan el vocabulario de la app. Las 20 **variadas** son "
                        "coloquiales, en inglés, con dos intenciones o piden una explicación. La referencia de cada "
                        "caso la calculan las mismas herramientas.")
            st.markdown("Las reglas por palabras clave aciertan las directas, pero no las variadas. Sin herramientas, "
                        "Haiku no tiene los datos.")
        st.dataframe(pd.DataFrame([{
            "Configuración": names.get(k, k), "Exactitud %": m["exactitud_pct"],
            "Con cifras sin respaldo %": m["respuestas_con_cifras_no_trazables_pct"],
            "Latencia s": m["latencia_media_s"], "Tokens de entrada": m["tokens_entrada_medios"],
            "Costo por consulta USD": m["costo_por_consulta_usd"]} for k, m in llm_metrics.items()]).style.format(
            {"Exactitud %": "{:.1f}", "Con cifras sin respaldo %": "{:.1f}", "Latencia s": "{:.2f}",
             "Costo por consulta USD": "{:.5f}"}), hide_index=True, width="stretch")
        st.caption("Detalle por caso en artifacts/llm/. Se reproduce con python -m src.llm_eval.")


# --------------------------------------------------------------------------
# 4. Reserva
# --------------------------------------------------------------------------
with tabs[3]:
    st.markdown(f"<p class='lead'>Para cada trayecto se reserva lo pronosticado más un margen. Un pronóstico más "
                "preciso necesita menos margen para cubrir la misma cantidad de trayectos.</p>", unsafe_allow_html=True)
    with st.expander("Supuestos económicos"):
        a1, a2, a3 = st.columns(3)
        a1.text_input("Moneda", key="currency")
        a1.number_input(f"Precio del combustible por litro", 0.01, 100.0, step=0.01, key="price",
                        help="EIA, gasolina regular Midwest, 26 de octubre de 2026: 4.269 USD por galón.")
        a2.number_input("Ahorro cuando la reserva alcanza", 0.0, 10000.0, step=5.0, key="saving_hit")
        a2.number_input("Recargo cuando no alcanza", 0.0, 10000.0, step=5.0, key="deficit_fixed")
        a3.slider("Prima por litro faltante, veces el precio", 0.0, 3.0, step=0.05, key="deficit_premium")
        a3.slider("Costo por litro sobrante, veces el precio", 0.0, 1.0, step=0.01, key="excess_frac")
    PRM = ValueParams(st.session_state["price"], st.session_state["saving_hit"], st.session_state["deficit_fixed"],
                      st.session_state["deficit_premium"], st.session_state["excess_frac"])

    target = st.slider("Trayectos que la reserva debe cubrir", 80.0, 99.9, 95.0, 0.1, format="%.1f %%")
    rows = []
    for m in METHODS:
        ratio_v = trips_val["L_real"] / trips_val[f"L_{m}"].clip(lower=1e-6)
        mreq = float(np.quantile(ratio_v, target / 100) - 1)
        R = trips_test[f"L_{m}"] * (1 + mreq)
        exc = float((R - trips_test["L_real"]).clip(lower=0).mean())
        rows.append({"Método": LABELS[m], "Margen necesario %": 100 * mreq,
                     "Cubiertos en prueba %": 100 * float((R >= trips_test["L_real"]).mean()),
                     "Sobrante por 1 000 trayectos L": 1000 * exc,
                     f"Combustible inmovilizado por 1 000 trayectos ({st.session_state['currency']})": 1000 * exc * PRM.price})
    st.dataframe(pd.DataFrame(rows).style.format(precision=1), hide_index=True, width="stretch")
    st.caption("El margen se elige en validación y se mide en prueba.")

    Av, At = trips_val["L_real"].to_numpy(), trips_test["L_real"].to_numpy()
    res = {}
    for m in METHODS:
        cv = value_curve(Av, trips_val[f"L_{m}"].to_numpy(), PRM)
        res[m] = outcome(At, trips_test[f"L_{m}"].to_numpy(), float(cv.loc[cv["value_per_trip"].idxmax(), "margin"]), PRM)
        res[m]["sigma"] = log_error_sigma(At, trips_test[f"L_{m}"].to_numpy())
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("##### Valor por trayecto según el margen")
        fig = go.Figure()
        for m in METHODS:
            cv = value_curve(At, trips_test[f"L_{m}"].to_numpy(), PRM)
            fig.add_trace(go.Scatter(x=100 * cv["margin"], y=cv["value_per_trip"], name=LABELS[m],
                                     line=dict(color=COLORS[m], width=2)))
        st.plotly_chart(style_fig(fig, 340, f"Valor por trayecto ({st.session_state['currency']})", "Margen %"),
                        width="stretch", config={"displayModeBar": False})
    with c2:
        st.markdown("##### Valor máximo según el error")
        ec = error_value_curve(At, PRM)
        fig = go.Figure(go.Scatter(x=100 * ec["sigma"], y=ec["value_per_trip"], name="Curva", line=dict(color=INK, width=1.5)))
        for m in METHODS:
            fig.add_trace(go.Scatter(x=[100 * res[m]["sigma"]], y=[float(np.interp(res[m]["sigma"], ec["sigma"], ec["value_per_trip"]))],
                                     mode="markers", name=LABELS[m], marker=dict(size=11, color=COLORS[m])))
        st.plotly_chart(style_fig(fig, 340, f"Valor por trayecto ({st.session_state['currency']})",
                                  "Error de pronóstico, desviación del log %"), width="stretch",
                        config={"displayModeBar": False})
    st.caption("Los parámetros son supuestos documentados en config.py. La app no afirma ahorros verificados.")


# --------------------------------------------------------------------------
# 5. Datos y método
# --------------------------------------------------------------------------
with tabs[4]:
    if quality:
        st.markdown(f"<p class='lead'>Vehicle Energy Dataset: {quality['ice_raw_records'] / 1e6:.1f} millones de "
                    f"registros OBD-II de vehículos de gasolina en Ann Arbor. Después de filtrar quedan "
                    f"{quality['vehicles_kept']} vehículos, {thousands(quality['trips_kept'])} trayectos y "
                    f"{thousands(quality['hours_kept'])} horas.</p>", unsafe_allow_html=True)
    st.image(str(ROOT / "app" / "assets" / "pipeline.png"), width="stretch")
    if quality:
        c1, c2 = st.columns(2, gap="large")
        names = {"speed": "Velocidad", "rpm": "RPM", "load": "Carga del motor", "maf": "Flujo de aire",
                 "fuel_rate": "Flujo de combustible medido", "oat": "Temperatura exterior"}
        c1.markdown("##### Registros con dato")
        c1.dataframe(pd.DataFrame({"Variable": [names[k] for k in quality["raw_coverage_frac"]],
                                   "%": [100 * v for v in quality["raw_coverage_frac"].values()]})
                     .style.format({"%": "{:.2f}"}), hide_index=True, width="stretch")
        c2.markdown("##### Cómo se estimó el consumo")
        c2.markdown("El flujo de combustible casi nunca viene medido. Se estima con el Algoritmo 1 del artículo de "
                    "VED a partir del flujo de aire y los ajustes de mezcla. Si no hay flujo de aire, se usa carga, "
                    "RPM y cilindrada.")
        sv = quality["split_vehicles"]
        c2.markdown(f"La división es por vehículo: {sv['train']} para entrenar, {sv['val']} para validar y "
                    f"{sv['test']} para probar. El escalado usa solo entrenamiento.")
    st.markdown("##### Cuándo no usarlo")
    st.markdown("En vehículos híbridos, eléctricos o diésel, en trayectos largos de carretera distintos a los de "
                "Ann Arbor, o para decisiones de seguridad del vehículo.")


# --------------------------------------------------------------------------
# 6. Subir trayecto
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
    st.markdown("<p class='lead'>Sube un CSV de un trayecto en formato VED para ver el pronóstico del GRU. Columnas "
                "mínimas: Timestamp(ms), Vehicle Speed[km/h], Engine RPM[RPM], Absolute Load[%] y MAF[g/sec] o "
                "Fuel Rate[L/hr].</p>", unsafe_allow_html=True)
    sample = ROOT / "app" / "sample_trip.csv"
    c1, c2, c3 = st.columns([1.2, 1, 1])
    up = c1.file_uploader("CSV del trayecto", type="csv")
    disp = c2.number_input("Cilindrada en litros, si no hay flujo de aire", 0.6, 8.0, 2.0, 0.1)
    hist_rate = c3.number_input("Promedio histórico del vehículo, L/h", 0.1, 50.0,
                                round(metrics["train_mean_rate_Lh"], 2), 0.1)
    if sample.exists():
        c1.download_button("Descargar un CSV de ejemplo", sample.read_bytes(), "sample_trip.csv", "text/csv",
                           type="tertiary")
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
            last = int((len(blk_u) - 1) * BLOCK_MIN)
            mu = st.slider("Minuto", 0, last, last, step=int(BLOCK_MIN), format="%d min", key="ub") if last > 0 else 0
            st.plotly_chart(trip_figure(ub, blk_u, int(mu // BLOCK_MIN), ["gru", "hist"]), width="stretch",
                            config={"displayModeBar": False})
            rows = [{"Método": LABELS[m], "Litros": blk_u[f"L_{m}"].sum(), f"Costo ({currency})": blk_u[f"L_{m}"].sum() * price}
                    for m in ["real", "gru", "hist"]]
            st.dataframe(pd.DataFrame(rows).style.format(precision=3), hide_index=True)
            st.caption("Medido es el consumo estimado con el Algoritmo 1 de VED a partir del flujo de aire.")
