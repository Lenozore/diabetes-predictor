"""
Diabetes Prediction — Flask backend.
Loads the artifacts produced by the training notebook (model_artifacts/) and
serves a REST API plus the front-end pages described in the NTCC report,
Section 7 (System Architecture and Web Application Development).
"""
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

from preprocessing import ZeroImputer, IQRCapper  # noqa: F401  (needed for joblib.load)

BASE_DIR = Path(__file__).resolve().parent
ART_DIR = BASE_DIR / "model_artifacts"

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Load trained artifacts once at startup (Section 9.1: avoid per-request I/O)
# ---------------------------------------------------------------------------
with open(ART_DIR / "model_metadata.json") as f:
    METADATA = json.load(f)

FEATURE_ORDER = METADATA["feature_order"]
ZERO_AS_MISSING = METADATA["zero_as_missing_cols"]
MODEL_TYPE = METADATA["model_type"]
MODEL_NAME = METADATA["selected_model"]

scaler = joblib.load(ART_DIR / "scaler.pkl")
imputer = joblib.load(ART_DIR / "imputer.pkl")
capper = joblib.load(ART_DIR / "capper.pkl")

if MODEL_TYPE == "keras":
    from tensorflow import keras
    model = keras.models.load_model(ART_DIR / "best_model.keras")

    def predict_proba(df):
        return model.predict(df.values, verbose=0).ravel()
else:
    model = joblib.load(ART_DIR / "best_model.pkl")

    def predict_proba(df):
        return model.predict_proba(df)[:, 1]

# Lazy SHAP explainer (built on first /explain call to keep startup fast)
_explainer = None


def get_explainer():
    global _explainer
    if _explainer is None:
        import shap
        if MODEL_TYPE == "sklearn" and hasattr(model, "get_booster"):
            _explainer = ("tree", shap.TreeExplainer(model))
        elif MODEL_TYPE == "sklearn" and hasattr(model, "tree_"):
            _explainer = ("tree", shap.TreeExplainer(model))
        elif MODEL_TYPE == "sklearn" and hasattr(model, "estimators_"):
            _explainer = ("tree", shap.TreeExplainer(model))
        else:
            # Linear / SVM / ANN fall back to a light KernelExplainer.
            # Built lazily against a small background sample stored at train time.
            background = np.zeros((1, len(FEATURE_ORDER)))
            fn = predict_proba if MODEL_TYPE != "keras" else predict_proba
            _explainer = ("kernel", shap.KernelExplainer(
                lambda x: predict_proba(pd.DataFrame(x, columns=FEATURE_ORDER)), background))
    return _explainer


# In-memory prediction log for the analytics dashboard. The report scopes a
# persistent database/logging layer as future work (Section 12), so this is
# intentionally session-only rather than backed by a real database.
PREDICTION_LOG = []


def build_features(payload):
    """Validate + preprocess one patient record. Returns (df_scaled, raw_df, error)."""
    try:
        row = {k: float(payload[k]) for k in FEATURE_ORDER}
    except (KeyError, TypeError, ValueError):
        return None, None, f"Missing or non-numeric field. Expected: {FEATURE_ORDER}"

    ranges = {
        "Pregnancies": (0, 20), "Glucose": (0, 300), "BloodPressure": (0, 200),
        "SkinThickness": (0, 100), "Insulin": (0, 900), "BMI": (0, 80),
        "DiabetesPedigreeFunction": (0, 3), "Age": (1, 120),
    }
    for k, (lo, hi) in ranges.items():
        if not (lo <= row[k] <= hi):
            return None, None, f"{k}={row[k]} is outside the physiologically plausible range [{lo}, {hi}]"

    raw_df = pd.DataFrame([row])[FEATURE_ORDER]
    df = imputer.transform(raw_df)
    df = capper.transform(df)
    scaled = pd.DataFrame(scaler.transform(df), columns=FEATURE_ORDER)
    return scaled, raw_df, None


@app.route("/")
def home():
    return render_template("index.html", model_name=MODEL_NAME, metrics=METADATA["test_metrics"])


@app.route("/predict-page")
def predict_page():
    return render_template("predict.html", feature_order=FEATURE_ORDER)


@app.route("/dashboard")
def dashboard():
    return render_template(
        "dashboard.html",
        all_metrics=METADATA["all_model_metrics"],
        shap_ranking=METADATA.get("shap_ranking") or {},
        selected_model=MODEL_NAME,
        log=list(reversed(PREDICTION_LOG[-25:])),
    )


@app.route("/about")
def about():
    return render_template("about.html", model_name=MODEL_NAME, metrics=METADATA["test_metrics"])





@app.route("/api/predict", methods=["POST"])
def api_predict():
    payload = request.get_json(force=True, silent=True) or {}
    scaled, raw_df, error = build_features(payload)
    if error:
        return jsonify({"error": error}), 400

    t0 = time.time()
    proba = float(predict_proba(scaled)[0])
    latency_ms = (time.time() - t0) * 1000
    pred = int(proba >= 0.5)

    record = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "input": raw_df.iloc[0].to_dict(),
        "prediction": pred,
        "probability": round(proba, 4),
    }
    PREDICTION_LOG.append(record)

    return jsonify({
        "prediction": pred,
        "label": "Diabetic risk indicated" if pred else "Non-diabetic risk profile",
        "probability": round(proba, 4),
        "model_used": MODEL_NAME,
        "latency_ms": round(latency_ms, 3),
        "disclaimer": "Screening estimate only. Not a medical diagnosis.",
    })


@app.route("/api/explain", methods=["POST"])
def api_explain():
    payload = request.get_json(force=True, silent=True) or {}
    scaled, raw_df, error = build_features(payload)
    if error:
        return jsonify({"error": error}), 400

    kind, explainer = get_explainer()
    if kind == "tree":
        sv = explainer.shap_values(scaled)
        sv = sv[1] if isinstance(sv, list) else sv
        sv = np.array(sv)[0]
    else:
        sv_full = explainer.shap_values(scaled.values, nsamples=100)
        sv = np.array(sv_full)[0] if not isinstance(sv_full, list) else np.array(sv_full[1])[0]

    contributions = sorted(
        zip(FEATURE_ORDER, sv.tolist()), key=lambda kv: abs(kv[1]), reverse=True
    )
    return jsonify({"contributions": contributions})


@app.route("/api/model-info")
def api_model_info():
    return jsonify(METADATA)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
