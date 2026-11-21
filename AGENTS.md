# heart-disease-prediction

Recall-first binary classifier on UCI Heart Disease (Cleveland, 303 rows): LR vs RF vs XGBoost, PostgreSQL run history, served by Streamlit and a React/Flask Vercel app. Educational demo, not a diagnostic tool.

## Commands
- Setup: `py -3.12 -m venv .venv`, `pip install -r requirements-dev.txt`, `cp .env.example .env`, `docker compose up -d` (postgres:16 on host port 5433)
- Pipeline: `python -m src.ingest` -> `python -m src.train` (~7 min) -> `python -m src.evaluate` (activates the winner) -> `python -m src.external`
- Run: `streamlit run app/streamlit_app.py`
- Check: `ruff check . && ruff format --check .` and `pytest -q -m "not db"` (what CI runs); plain `pytest -q` adds DB tests, which skip if Postgres is down
- Vercel app, in `vercel_app/`: `npm ci && npm run check:models && npm run build`; local preview `python vercel_app/app.py` (port 5000) or `npm run dev`

## Map
| Path | What lives there |
|---|---|
| `src/` | config (SEED, RECALL_TARGET, PRECISION_FLOOR), db (SQLAlchemy ORM), ingest, data, features, models, train, evaluate, robustness, external, inference |
| `app/streamlit_app.py` | Streamlit form; falls back to `models/metadata.json` without Postgres |
| `vercel_app/` | Flask API (`GET /api/model`, `POST /api/predict`) + React UI; `models/` and `public/` are committed copies |
| `models/` | committed joblib artifacts + `metadata.json`, so the apps run on a fresh clone |
| `reports/` | `model_comparison.md`, `external_validation.md`, `figures/` |
| `notebooks/01_eda.ipynb` | exploration only; nothing in `src/` depends on it |

## Non-obvious facts
- Recall first: a missed sick patient costs more than a false alarm. Never select or report a model on accuracy, and never default to a 0.5 threshold. The threshold is the highest cut-off with OOF train recall >= `RECALL_TARGET`.
- Split before any fit (stratified 80/20, seed 42). Model selection and threshold tuning use CV on train only; the test set is used once, in `evaluate.py`.
- Preprocessing lives inside the saved Pipeline. The apps must never re-implement encoding or scaling.
- LR is the baseline to beat: a tree model ships only if its precision at recall >= target (5x5 repeated CV, `src/robustness.py`) beats LR's by more than one LR std. Ties go to LR.
- The displayed score is Platt-recalibrated on train OOF predictions. It is monotone, so it never changes who is flagged.
- Leakage alarm: test accuracy above 0.92 or recall of 1.00 means investigate (duplicates, wrong file) before reporting.
- The test set is ~61 rows; one miss moves recall about 3.6 points. Say so when reporting.
- Data: UCI file only. Kaggle `heart.csv` variants have an inverted target and fake NaN codes (`ca=4`, `thal=0`); the 1,025-row copy repeats patients. `target = (num > 0)`, 139/303 positive, missing only `ca` (4) and `thal` (2). `validate()` raises on unexpected codes; never coerce silently.
- After ingest, code reads Postgres, never the CSV. Ingest is idempotent. Each `model_runs` row stores `data_hash`; exactly one `is_active` run, switched in a transaction. Model binaries stay on disk; only their paths go in the DB. ORM or bound parameters only.
- `predictions` is written only when `LOG_PREDICTIONS=true`, and never stores identifiers.
- UI wording: show probability, threshold and "Elevated risk — recommend clinical follow-up" or "Lower risk". Never say "has/does not have heart disease". Keep the disclaimer (303 patients, one clinic, 1980s data, 68% male).
- After an evaluated model change, run `node vercel_app/scripts/sync-models.mjs` and commit the copies (CI checks byte equality). After frontend changes, run `npm run build` in `vercel_app/` and commit `public/` (CI fails on any diff). Pushes to `main` auto-deploy to Vercel.
