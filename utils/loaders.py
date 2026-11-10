"""
Carga de datos y modelos con caché, predicción de disrupciones y corrida de escenarios.

Si faltan los CSV o los modelos (por ejemplo, la primera vez en Streamlit Cloud),
se generan solos con los mismos scripts de data/ y ml/.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from optimization import clustering, milp
from utils import theme as th

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
TABLES = ["flowers", "farms", "hubs", "nodes", "lanes", "markets", "chains", "demand", "arcs", "disruptions_history",
          "shipments_history"]


def _ensure_data() -> None:
    if not all((DATA_DIR / f"{t}.csv").exists() for t in TABLES):
        from data import generate_data
        generate_data.main()


@st.cache_data(show_spinner=False)
def _read_all() -> dict:
    _ensure_data()
    data = {t: pd.read_csv(DATA_DIR / f"{t}.csv", keep_default_na=False, na_values=[""]) for t in TABLES}
    data["shipments_history"]["date"] = pd.to_datetime(data["shipments_history"]["date"])
    data["nodes"]["market"] = data["nodes"]["market"].fillna("")
    data["chains"]["via"] = data["chains"]["via"].fillna("")
    return data


def load_data() -> dict:
    """Todos los CSV en un diccionario de DataFrames (y las etiquetas cortas de los nodos en theme)."""
    data = _read_all()
    if not th.NODE_LABEL:
        n = data["nodes"]
        th.NODE_LABEL.update(dict(zip(n.node_id, n.short)))
        d = n[n.type == "destination"]
        th.DEST_LABEL.update({i: f"{s} ({m})" for i, s, m in zip(d.node_id, d.short, d.market)})
    return data


def market_info(code: str) -> dict:
    """Fila del mercado (nombre, rol, pedido semanal...) como diccionario."""
    m = load_data()["markets"].set_index("market")
    return {"market": code, **m.loc[code].to_dict()}


def market_codes() -> list[str]:
    return load_data()["markets"].market.tolist()


@st.cache_resource(show_spinner=False)
def load_models() -> dict:
    """Modelos XGBoost guardados con joblib. Si no existen o no cargan, se vuelven a entrenar."""
    from ml import train_xgboost as tx
    _ensure_data()
    try:
        models = {k: joblib.load(tx.model_path(k)) for k in tx.DISRUPTIONS}
        if not (MODELS_DIR / "metrics.json").exists():
            raise FileNotFoundError
    except Exception:
        tx.train_all(verbose=False)
        models = {k: joblib.load(tx.model_path(k)) for k in tx.DISRUPTIONS}
    return models


@st.cache_data(show_spinner=False)
def load_metrics() -> dict:
    load_models()
    return json.loads((MODELS_DIR / "metrics.json").read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def test_dates() -> tuple[str, str, str]:
    """Rango de fechas que se pueden predecir (los 12 meses que el modelo no vio) y fecha de ejemplo."""
    m = load_metrics()
    d = load_data()["disruptions_history"]
    d = d[d.date >= m["test_start"]]
    return d.date.min(), d.date.max(), m.get("showcase_date", d.date.min())


@st.cache_data(show_spinner=False)
def predict(date: str) -> dict:
    """Probabilidad de cada disrupción para la fecha dada, con las variables de ese día."""
    from ml.train_xgboost import DISRUPTIONS, FEATURES
    hist = load_data()["disruptions_history"]
    row = hist[hist.date == date]
    if row.empty:
        row = hist.tail(1)
    models = load_models()
    out = {}
    for key, info in DISRUPTIONS.items():
        out[key] = dict(prob=float(models[key].predict_proba(row[FEATURES])[0, 1]), actual=int(row[info["label"]].iloc[0]),
                        title=info["title"], effect=info["effect"])
    return dict(probs=out, features=row[FEATURES].iloc[0].to_dict())


@st.cache_data(show_spinner=False)
def clusters() -> dict:
    """k-medoids (PAM) por región, con k por silueta, y los medoides como sitios candidatos."""
    data = load_data()
    res = clustering.cluster_farms(data["farms"])
    res["medoid_sites"] = clustering.medoids_as_hub_candidates(res["medoids"], data["hubs"])
    res["all_medoid_sites"] = clustering.medoids_as_hub_candidates(res["medoids"], data["hubs"], min_gap_km=0.0)
    return res


@st.cache_data(show_spinner=False)
def _run(sc_items: tuple, active: tuple) -> dict:
    sc = milp.Scenario(**dict(sc_items))
    cl = clusters()
    return milp.run_scenario(load_data(), sc, frozenset(active), cl["medoid_sites"], cl["all_medoid_sites"])


def run_scenario(sc: milp.Scenario, active: set) -> dict:
    """Resuelve el escenario (con caché: repetir los mismos parámetros es instantáneo)."""
    return _run(tuple(sorted(asdict(sc).items())), tuple(sorted(active)))


# ----------------------------------------------------------------------------
# Estado compartido entre páginas
# ----------------------------------------------------------------------------
DEFAULT_PARAMS = dict(
    market="NL", n_range=(1, 5), max_farm_hub_km=30, min_hub_gap_km=5, max_hub_capacity=5000, candidate_source="both",
    carbon_price=80, min_vase_life=9.0, quality_pct=6.0, flowers=["rose", "carnation", "chrysanthemum"],
    threshold=0.50, manual=[],
)


def get_params() -> dict:
    if "params" not in st.session_state:
        p = dict(DEFAULT_PARAMS)
        p["date"] = test_dates()[2]
        st.session_state["params"] = p
    for k, v in DEFAULT_PARAMS.items():           # sesiones abiertas antes de agregar un parámetro nuevo
        st.session_state["params"].setdefault(k, v)
    return st.session_state["params"]


def scenario_from(p: dict) -> milp.Scenario:
    return milp.Scenario(
        n_min=int(p["n_range"][0]), n_max=int(p["n_range"][1]), max_farm_hub_km=float(p["max_farm_hub_km"]),
        min_hub_gap_km=float(p["min_hub_gap_km"]), max_hub_capacity=int(p["max_hub_capacity"]),
        carbon_price=float(p["carbon_price"]), min_vase_life=float(p["min_vase_life"]), quality_pct=float(p["quality_pct"]),
        flowers=tuple(p["flowers"]), candidate_source=p["candidate_source"], market=p["market"],
    )


def active_disruptions(p: dict) -> tuple[set, dict]:
    """Disrupciones activas = las que XGBoost predice sobre el umbral + las que el usuario fuerza a mano."""
    pred = predict(p["date"])
    active = {k for k, v in pred["probs"].items() if v["prob"] >= p["threshold"]} | set(p["manual"])
    return active, pred


def compute(p: dict) -> dict:
    """Corre todo para los parámetros dados y deja el resultado en la sesión."""
    active, pred = active_disruptions(p)
    res = run_scenario(scenario_from(p), active)
    out = dict(params=dict(p), pred=pred, active=sorted(active), res=res)
    st.session_state["results"] = out
    return out


def get_results() -> dict:
    """Resultado actual; si todavía no se ha corrido nada, corre el escenario por defecto."""
    if "results" not in st.session_state:
        with st.spinner("Solving the default scenario…"):
            compute(get_params())
        st.session_state["fresh"] = True
    return st.session_state["results"]
