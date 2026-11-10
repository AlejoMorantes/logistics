"""
Entrena un clasificador XGBoost binario por cada tipo de disrupción.

Uso:
    python ml/train_xgboost.py

- Split temporal: los primeros 2 años entrenan y los últimos 12 meses prueban
  (no se mezclan fechas, porque en la vida real solo se conoce el pasado).
- Reporta AUC, matriz de confusión e importancia de variables.
- Guarda cada modelo con joblib en models/ y las métricas en models/metrics.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATA_PATH = ROOT / "data" / "disruptions_history.csv"
MODELS_DIR = ROOT / "models"
TEST_START = "2025-11-01"

# Tipos de disrupción: columna etiqueta y qué parte de la red cierran
DISRUPTIONS = {
    "fog_bog": dict(label="disrupted_fog_bog", title="Fog at El Dorado (BOG)",
                    effect="Every flight leaving Bogotá is grounded"),
    "road_strike": dict(label="disrupted_road_strike", title="Road strike (Bogotá–Caribbean corridor)",
                        effect="Trucks from Sabana hubs cannot reach the ports"),
    "port_congestion": dict(label="disrupted_port_congestion", title="Port congestion at Cartagena",
                            effect="Every sea lane leaving Cartagena is closed"),
    "atlantic_storm": dict(label="disrupted_atlantic_storm", title="Atlantic storm",
                           effect="Every transatlantic sea lane is closed"),
}

# Variables que se conocen la víspera (pronóstico de mañana + lo que pasó ayer)
FEATURES = [
    "month", "day_of_week", "peak_season",
    "rain_bog_mm", "humidity_bog_pct", "temp_min_bog_c", "wind_bog_kmh",
    "strike_announced", "fuel_price_change_pct",
    "port_yard_occupancy_pct", "vessels_waiting_ctg", "rain_caribbean_mm",
    "wind_caribbean_kmh", "wave_height_atl_m", "pressure_anomaly_hpa",
    "delay_prev_air_min", "delay_prev_sea_h",
    "fog_prev_day", "strike_prev_day", "congestion_prev_day", "storm_prev_day",
]


def model_path(key: str) -> Path:
    return MODELS_DIR / f"xgb_{key}.joblib"


def train_all(verbose: bool = True) -> dict:
    df = pd.read_csv(DATA_PATH)
    train, test = df[df.date < TEST_START], df[df.date >= TEST_START]
    MODELS_DIR.mkdir(exist_ok=True)
    metrics = {"features": FEATURES, "test_start": TEST_START, "n_train": len(train), "n_test": len(test), "models": {}}
    preds = pd.DataFrame({"date": test.date.values})

    for key, info in DISRUPTIONS.items():
        y_train, y_test = train[info["label"]], test[info["label"]]
        # Las disrupciones son pocas: se da más peso a la clase positiva
        pos_weight = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
        model = XGBClassifier(
            n_estimators=180, max_depth=3, learning_rate=0.06, subsample=0.9, colsample_bytree=0.9,
            min_child_weight=2, reg_lambda=1.5, scale_pos_weight=min(pos_weight, 4.0),
            eval_metric="logloss", random_state=42, n_jobs=2,
        )
        model.fit(train[FEATURES], y_train)
        proba = model.predict_proba(test[FEATURES])[:, 1]
        auc = float(roc_auc_score(y_test, proba))
        cm = confusion_matrix(y_test, (proba >= 0.5).astype(int), labels=[0, 1])
        fpr, tpr, _ = roc_curve(y_test, proba)
        importance = dict(sorted(zip(FEATURES, map(float, model.feature_importances_)), key=lambda kv: -kv[1]))

        joblib.dump(model, model_path(key))
        preds[f"p_{key}"] = np.round(proba, 4)
        preds[f"actual_{key}"] = y_test.values
        metrics["models"][key] = dict(
            title=info["title"], effect=info["effect"], auc=round(auc, 4),
            base_rate_train=round(float(y_train.mean()), 4), base_rate_test=round(float(y_test.mean()), 4),
            confusion_matrix=cm.tolist(),  # [[TN, FP], [FN, TP]] con umbral 0.5
            roc=dict(fpr=[round(float(v), 4) for v in fpr], tpr=[round(float(v), 4) for v in tpr]),
            importance=importance,
        )
        if verbose:
            tn, fp, fn, tp = cm.ravel()
            top = ", ".join(f"{k} ({v:.2f})" for k, v in list(importance.items())[:3])
            print(f"{key:16s} AUC={auc:.3f}  TN={tn} FP={fp} FN={fn} TP={tp}  top: {top}")

    # Fecha de ejemplo para abrir la app: congestión en Cartagena clara y lo demás tranquilo
    others = preds[[c for c in preds.columns if c.startswith("p_") and c != "p_port_congestion"]].max(axis=1)
    cand = preds[(preds.p_port_congestion >= 0.70) & (others < 0.25)]
    metrics["showcase_date"] = str(cand.date.iloc[0]) if len(cand) else str(preds.date.iloc[0])

    preds.to_csv(MODELS_DIR / "test_predictions.csv", index=False)
    (MODELS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=1), encoding="utf-8")
    if verbose:
        print("Fecha de ejemplo:", metrics["showcase_date"])
    return metrics


if __name__ == "__main__":
    train_all()
