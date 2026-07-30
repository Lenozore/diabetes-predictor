# Diabetes Prediction — Web App (Section 7 of the report)

Flask backend + HTML/CSS/JS frontend serving the model trained in
`diabetes_prediction_training.ipynb`. Verified working locally (dev server
and gunicorn) before hand-off.

## What's in here

```
app.py                 Flask app: page routes + /api/predict + /api/explain
preprocessing.py        ZeroImputer / IQRCapper — must match the notebook exactly
model_artifacts/        best_model.pkl, scaler.pkl, imputer.pkl, capper.pkl, model_metadata.json
templates/              Home, screening form, dashboard, about, contact
static/                 CSS + JS (gauge visualization, Chart.js dashboard)
requirements.txt
Procfile                for Render/Railway
```

## Run it locally

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
python3 app.py
```

Open http://127.0.0.1:5000. The dev server has the debug reloader on — fine
for local work, turn it off (`debug=False`) before deploying anywhere public.

## Re-training / swapping the model

If you re-run the notebook and a different model wins the comparison
(the notebook prints this — see `PRIORITIZE_RECALL` if you want to select
by recall instead of ROC-AUC), just overwrite the five files in
`model_artifacts/` with the notebook's fresh output. `app.py` reads
`model_metadata.json` to know which model type it's loading — no code
changes needed, *unless* the ANN/DNN wins, in which case uncomment the
`tensorflow` line in `requirements.txt`.

## Deploying (report Section 9.2 names Render, Railway, PythonAnywhere)

### Render (free tier friendly)
1. Push this folder to a GitHub repo.
2. Render → New → Web Service → connect the repo.
3. Build command: `pip install -r requirements.txt`
4. Start command: `gunicorn app:app` (already in the `Procfile`, Render
   picks it up automatically).
5. Deploy, then open the given `.onrender.com` URL.

### Railway
1. Push to GitHub, then Railway → New Project → Deploy from repo.
2. Railway auto-detects the `Procfile`. No extra config needed.

### PythonAnywhere
1. Upload this folder (or `git clone` it) into your PythonAnywhere account.
2. Web tab → Add a new web app → Flask → point the WSGI file at `app.app`.
3. Set the working directory to this folder so `model_artifacts/` resolves.

## A few things to check before treating this as production

- `PREDICTION_LOG` is an in-memory Python list — it resets on every
  restart/redeploy and isn't shared across gunicorn workers. The report's
  own Future Scope (Section 12) lists adding a real database as later work;
  this is that gap, left as-is rather than hidden.
- Input validation checks physiological ranges but this is still a
  screening demo, not a medical device — the UI says so on every page.
- If you move hosts, re-check the SHAP explainer choice in `get_explainer()`
  — it assumes a tree-based model; if you swap in Logistic Regression or
  SVM, it falls back to a slower `KernelExplainer`.
