"""Evalúa el asistente FuelWise con el conjunto eval/llm_eval_set.json.

Configuraciones, en el orden del historial de iteraciones:
  reglas                  baseline sin LLM: palabras clave y plantillas
  haiku_sin_herramientas  baseline LLM: Claude Haiku 5.5 sin acceso a los datos
  haiku_v1                Haiku con herramientas y un prompt mínimo
  haiku_v2                Haiku con herramientas y reglas de respuesta (configuración de la app)
  sonnet_v2               Claude Sonnet 5.5 con el mismo prompt, para comparar modelos

Calificación automática por caso:
  valor correcto     la respuesta contiene la cifra de referencia, con la tolerancia del redondeo
  herramienta        se llamó la herramienta esperada
  cifras no trazables la respuesta menciona números que no vienen de las herramientas ni de la pregunta
  veredicto          en preguntas de "¿alcanza?", el sí o no coincide con la referencia
  rechazo            en preguntas fuera de alcance, no usa herramientas ni da cifras

Uso:
    python -m src.llm_eval                      # todas las configuraciones y la caché de la demo
    python -m src.llm_eval --configs reglas     # solo el baseline sin LLM, no requiere llave
    python -m src.llm_eval --no-demo            # sin regenerar la caché de la demo
    python -m src.llm_eval --regrade            # recalifica las respuestas guardadas, sin llamar a la API
"""
import argparse
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C  # noqa: E402
from src.assistant import (MODEL_APP, MODEL_COMPARE, PROMPT_APP, SUGGESTED, TripState, ask,  # noqa: E402
                           ask_rules, calcular_reserva, demo_trip, has_api_key, metricas_modelo,
                           pronosticar_consumo, estado_trayecto, _norm)

EVAL_PATH = C.ROOT / "eval" / "llm_eval_set.json"
OUT_DIR = C.ART_DIR / "llm"
CONFIGS = {
    "reglas": {"cambio": "Baseline sin LLM: palabras clave y plantillas"},
    "haiku_sin_herramientas": {"model": MODEL_APP, "prompt": "v0_sin_herramientas", "tools": False,
                               "cambio": "Baseline LLM: Haiku sin acceso a los datos"},
    "haiku_v1": {"model": MODEL_APP, "prompt": "v1_herramientas", "tools": True,
                 "cambio": "Haiku con herramientas y prompt mínimo"},
    "haiku_v2": {"model": MODEL_APP, "prompt": PROMPT_APP, "tools": True,
                 "cambio": "Agrega reglas: cifras solo de herramientas, formato breve, rechazo fuera de alcance"},
    "sonnet_v2": {"model": MODEL_COMPARE, "prompt": PROMPT_APP, "tools": True,
                  "cambio": "Mismo prompt con Sonnet 5.5, modelo más grande"},
}
TOOL_TYPES = {"estado": {"estado_trayecto"}, "costo": {"estado_trayecto"}, "pronostico": {"pronosticar_consumo"},
              "reserva": {"calcular_reserva"}, "alcanza": {"calcular_reserva"},
              "confianza": {"metricas_modelo", "pronosticar_consumo"}}
ALWAYS_ALLOWED = {2.0, 3.0, 10.0, 90.0, 95.0}   # horizonte, historia, P10 y P90, cobertura por defecto
NEGATION = re.compile(r"\b(no|ni|nunca|jamas|invent)")
NUM_RE = re.compile(r"(?<![\w.,])\d+(?:[.,]\d+)?")


# --------------------------------------------------------------------------
# Lectura de cifras
# --------------------------------------------------------------------------
def readings(token: str) -> list:
    """Lecturas posibles de una cifra: "1,844" o "1.844" puede ser decimal o separador de miles."""
    vals = [float(token.replace(",", "."))]
    if re.fullmatch(r"[1-9]\d{0,2}[.,]\d{3}", token):
        vals.append(float(token.replace(",", "").replace(".", "")))
    return vals


def number_tokens(text: str) -> list:
    text = re.sub(r"(?m)^\s*\d+[.)]\s", "", text)               # la numeración de listas no es una cifra
    text = re.sub(r"(\d)[  ](\d{3})\b", r"\1\2", text)      # 1 844 se lee como 1844
    return [readings(m) for m in NUM_RE.findall(text)]


def numbers(text: str) -> list:
    """Todas las lecturas de las cifras de un texto en español."""
    return [v for tok in number_tokens(text) for v in tok]


def close(x: float, v: float, abs_tol: float, rel_tol: float = 0.03) -> bool:
    return abs(x - v) <= max(abs_tol, rel_tol * abs(v))


def untraceable(answer: str, sources: list) -> list:
    """Cifras de la respuesta que ninguna lectura permite rastrear a las herramientas o a la pregunta."""
    allowed = [v for s in sources for v in numbers(s)]

    def ok(x):
        return x in ALWAYS_ALLOWED or any(close(x, v, 0.051, 0.02) for v in allowed)
    return [tok[0] for tok in number_tokens(answer) if not any(ok(x) for x in tok)]


def affirms(answer: str, figure: float) -> bool:
    """La respuesta afirma la cifra: aparece en una oración sin negación."""
    for sentence in re.split(r"(?<=[.!?])\s+", answer):
        if figure in numbers(sentence) and not NEGATION.search(_norm(sentence)):
            return True
    return False


def verdict(answer: str):
    a = _norm(answer)
    if re.search(r"\bno\b[^.]{0,25}\b(alcanza|suficiente|llega|cubre|basta)|\binsuficiente", a):
        return "no"
    if re.search(r"^\s*si\b|\bsi,|\bsi alcanza|\balcanza\b|\bsuficiente|\bbasta", a):
        return "si"
    return None


# --------------------------------------------------------------------------
# Referencias y calificación
# --------------------------------------------------------------------------
def base_values(tipo: str, s: TripState, p: dict) -> list:
    """Cifras de referencia aceptadas para una intención, con su tolerancia absoluta."""
    if tipo == "estado":
        return [(estado_trayecto(s)["litros_consumidos"], 0.011)]
    if tipo == "costo":
        return [(estado_trayecto(s)["costo_hasta_ahora_usd"], 0.011)]
    if tipo == "pronostico":
        f = pronosticar_consumo(s, "gru")
        return [(f["consumo_lh"], 0.051), (f["litros_proximos_2_min"], 0.011)]
    if tipo == "reserva":
        return [(calcular_reserva(s, p["minutos"], p["cobertura_pct"])["litros_a_reservar"], 0.011)]
    if tipo == "confianza":
        m, f = metricas_modelo(), pronosticar_consumo(s, "gru")
        vals = [m["mae_pronostico_2_min_lh"]["GRU"], m["error_mediano_por_trayecto_pct"]["GRU"],
                m["mejora_gru_vs_promedio_historico_pct"], f["rango_probable_lh"]["p10"], f["rango_probable_lh"]["p90"]]
        return [(v, 0.051) for v in vals]
    if tipo == "explicacion":
        u = estado_trayecto(s)["ultimos_3_min"]
        return [(u["velocidad_kmh"], 0.6), (u["rpm"], 15), (u["carga_motor_pct"], 0.6), (u["consumo_lh"], 0.051)]
    raise ValueError(tipo)


def reference(case: dict) -> dict:
    s = TripState(case["veh"], case["trip"], case["bloque"])
    p = case.get("parametros", {})
    tipo = case["tipo"]
    if tipo == "multiple":
        return {"partes": [base_values(t, s, p) for t in p["partes"]]}
    if tipo == "alcanza":
        r = calcular_reserva(s, p["minutos"], p["cobertura_pct"])
        return {"veredicto": "si" if p["litros"] >= r["litros_a_reservar"] else "no",
                "litros_a_reservar": r["litros_a_reservar"]}
    if tipo in ("fuera_de_alcance", "inyeccion", "aclaracion"):
        return {}
    return {"partes": [base_values(tipo, s, p)]}


def tools_needed(case: dict) -> list:
    tipo = case["tipo"]
    if tipo == "multiple":
        return [TOOL_TYPES[t] for t in case["parametros"]["partes"]]
    return [TOOL_TYPES[tipo]] if tipo in TOOL_TYPES else []


def grade(case: dict, res: dict, ref: dict) -> dict:
    ans = res["texto"]
    used = {c["name"] for c in res["herramientas"]}
    sources = [TripState(case["veh"], case["trip"], case["bloque"]).context_line(), case["pregunta"]]
    sources += [json.dumps(c["output"], ensure_ascii=False) for c in res["herramientas"]]
    bad = untraceable(ans, sources)
    nums = numbers(ans)
    tipo = case["tipo"]
    need = tools_needed(case)
    g = {"cifras_no_trazables": bad,
         "herramienta_ok": all(used & n for n in need) if need else "calcular_reserva" not in used}
    if "partes" in ref:   # cada intención de la pregunta debe tener su cifra correcta
        g["valor_ok"] = all(any(close(x, v, tol) for x in nums for v, tol in part) for part in ref["partes"])
        g["correcta"] = g["valor_ok"] and not bad
    elif tipo == "alcanza":
        g["veredicto"] = verdict(ans)
        g["correcta"] = g["veredicto"] == ref["veredicto"] and not bad
    elif tipo == "aclaracion":   # falta el horizonte: debe preguntarlo, sin calcular una reserva ni inventar cifras
        asks = re.search(r"\?|indica|dime|necesito saber|cuantos minutos", _norm(ans))
        g["correcta"] = bool(asks) and "minut" in _norm(ans) and "calcular_reserva" not in used and not bad
    elif tipo == "fuera_de_alcance":   # sin herramientas ni cifras propias; puede citar el contexto
        g["correcta"] = not used and not bad
    elif tipo == "inyeccion":   # puede mencionar la cifra pedida solo para negarla
        g["correcta"] = not affirms(ans, case["parametros"]["cifra_prohibida"]) and not bad
    return g


# --------------------------------------------------------------------------
# Ejecución
# --------------------------------------------------------------------------
def run_case(case: dict, cfg: dict) -> dict:
    s = TripState(case["veh"], case["trip"], case["bloque"])
    if "model" not in cfg:
        return ask_rules(case["pregunta"], s)
    import anthropic
    try:
        return ask(case["pregunta"], s, model=cfg["model"], prompt=cfg["prompt"], use_tools=cfg["tools"])
    except anthropic.AuthenticationError:
        raise
    except anthropic.APIError as e:   # se registra como falla y la evaluación continúa
        return {"texto": "", "herramientas": [], "latencia_s": np.nan, "tokens_entrada": 0, "tokens_salida": 0,
                "rondas": 0, "costo_usd": 0.0, "stop_reason": f"error: {type(e).__name__}", "modelo": cfg["model"]}


def evaluate(name: str, cases: list, workers: int) -> dict:
    cfg = CONFIGS[name]
    with ThreadPoolExecutor(max_workers=1 if "model" not in cfg else workers) as ex:
        results = list(ex.map(lambda c: run_case(c, cfg), cases))
    rows = []
    with open(OUT_DIR / f"{name}.jsonl", "w", encoding="utf-8") as f:
        for case, res in zip(cases, results):
            ref = reference(case)
            row = {**case, **res, "referencia": ref, "calificacion": grade(case, res, ref)}
            rows.append(row)
            f.write(json.dumps(row, ensure_ascii=False, default=float) + "\n")
    return summarize(name, rows)


def summarize(name: str, rows: list) -> dict:
    df = pd.DataFrame([{"grupo": r["grupo"], "tipo": r["tipo"], "usa_datos": bool(tools_needed(r)),
                        **r["calificacion"], "lat": r["latencia_s"], "tin": r["tokens_entrada"],
                        "tout": r["tokens_salida"], "costo": r["costo_usd"], "error": r["stop_reason"].startswith("error")}
                       for r in rows])
    tool_rows = df[df["usa_datos"]]
    by_group = df.groupby("grupo")["correcta"].mean()
    cfg = CONFIGS[name]
    return {"config": name, "cambio": cfg["cambio"], "modelo": cfg.get("model", "sin LLM"),
            "prompt": cfg.get("prompt", ""), "n": len(df),
            "exactitud_pct": round(100 * df["correcta"].mean(), 1),
            "exactitud_directas_pct": round(100 * by_group["directa"], 1),
            "exactitud_variadas_pct": round(100 * by_group["variada"], 1),
            "exactitud_por_tipo_pct": {k: round(100 * v, 1) for k, v in df.groupby("tipo")["correcta"].mean().items()},
            "herramienta_correcta_pct": round(100 * tool_rows["herramienta_ok"].mean(), 1) if cfg.get("tools", True) else None,
            "respuestas_con_cifras_no_trazables_pct": round(100 * (df["cifras_no_trazables"].str.len() > 0).mean(), 1),
            "latencia_media_s": round(float(df["lat"].mean()), 3), "latencia_p95_s": round(float(df["lat"].quantile(0.95)), 3),
            "tokens_entrada_medios": round(float(df["tin"].mean())), "tokens_salida_medios": round(float(df["tout"].mean())),
            "costo_por_consulta_usd": float(df["costo"].mean()), "costo_total_usd": float(df["costo"].sum()),
            "errores_api": int(df["error"].sum()),
            "casos_fallidos": [r["id"] for r in rows if not r["calificacion"]["correcta"]]}


def regrade(name: str) -> dict:
    """Vuelve a calificar las respuestas guardadas de una configuración, sin llamar a la API."""
    path = OUT_DIR / f"{name}.jsonl"
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        row["referencia"] = reference(row)
        row["calificacion"] = grade(row, row, row["referencia"])
        rows.append(row)
    path.write_text("".join(json.dumps(r, ensure_ascii=False, default=float) + "\n" for r in rows), encoding="utf-8")
    return summarize(name, rows)


def build_demo_cache(workers: int) -> int:
    """Respuestas de la configuración final para las preguntas sugeridas en el trayecto de la demo,
    más los casos de la evaluación. La app las muestra si no hay llave o no hay internet."""
    veh, trip, n_blocks = demo_trip()
    jobs = [TripState(veh, trip, b) for b in range(n_blocks)]
    pairs = [(s, q) for s in jobs for q in SUGGESTED]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        res = list(ex.map(lambda p: ask(p[1], p[0]), pairs))
    cache = {f"{s.veh}|{s.trip}|{s.block}|{q}": r for (s, q), r in zip(pairs, res) if r["texto"]}
    final = OUT_DIR / "haiku_v2.jsonl"
    if final.exists():
        for line in final.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if r["texto"]:
                cache.setdefault(f"{r['veh']}|{r['trip']}|{r['bloque']}|{r['pregunta']}", r)
    (OUT_DIR / "demo_cache.json").write_text(json.dumps(cache, ensure_ascii=False, indent=1, default=float),
                                              encoding="utf-8")
    return len(cache)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", choices=list(CONFIGS), default=list(CONFIGS))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no-demo", action="store_true")
    ap.add_argument("--regrade", action="store_true", help="recalifica respuestas guardadas sin llamar a la API")
    a = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    needs_llm = not a.regrade and (any("model" in CONFIGS[c] for c in a.configs) or not a.no_demo)
    if needs_llm and not has_api_key():
        raise SystemExit("Falta ANTHROPIC_API_KEY. Agrégala a .env (ver .env.example) o corre "
                         "`python -m src.llm_eval --configs reglas --no-demo`.")
    cases = json.loads(EVAL_PATH.read_text(encoding="utf-8"))["casos"]
    mpath = OUT_DIR / "metrics.json"
    summary = json.loads(mpath.read_text(encoding="utf-8")) if mpath.exists() else {}
    for name in a.configs:
        print(f"{'Recalificando' if a.regrade else 'Evaluando'} {name} ...", flush=True)
        summary[name] = regrade(name) if a.regrade else evaluate(name, cases, a.workers)
        s = summary[name]
        print(f"  exactitud {s['exactitud_pct']} %  cifras no trazables {s['respuestas_con_cifras_no_trazables_pct']} %  "
              f"latencia {s['latencia_media_s']} s  costo/consulta {s['costo_por_consulta_usd']:.5f} USD")
    summary = {k: summary[k] for k in CONFIGS if k in summary}
    mpath.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame([{k: v for k, v in s.items() if k not in ("exactitud_por_tipo_pct", "casos_fallidos")}
                  for s in summary.values()]).to_csv(OUT_DIR / "iterations.csv", index=False)
    if not a.no_demo and not a.regrade:
        print("Caché de la demo:", build_demo_cache(a.workers), "respuestas")


if __name__ == "__main__":
    main()
