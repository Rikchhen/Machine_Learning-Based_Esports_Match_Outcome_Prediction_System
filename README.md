# VCT Match Predictor

Predicting the outcome of professional Valorant (VCT) matches from pre-match
player statistics, with a Django web interface for exploring the models.

Trained on **12,558 matches** across VCT 2021–2026 (VLR.gg data), using an
ensemble of Logistic Regression, Random Forest and XGBoost over 11 team-level
differential features.

---

## Quick start

The trained artefacts are committed under `ml/saved_models/`, so the web app
runs without the raw dataset:

```bash
python -m venv pred_env
pred_env\Scripts\activate          # Windows
# source pred_env/bin/activate     # macOS / Linux

pip install -r requirements.txt
python manage.py migrate
python manage.py runserver
```

Then open <http://127.0.0.1:8000/>.

| Page | What it shows |
| --- | --- |
| `/` | Pick two teams from the dropdowns, get an ensemble win probability plus a per-stat breakdown |
| `/analysis/` | Test-set metrics, ROC and calibration curves, SHAP and feature importances |
| `/teams/` | Every team in the dataset, with a search filter |

A chosen matchup is shareable as a URL: `/?team_a=Sentinels&team_b=Paper%20Rex`.
The team cards on `/teams/` link to the predictor this way.

A single prediction is also available from the command line:

```bash
python -m ml.predict --team_a "Sentinels" --team_b "Paper Rex"
```

---

## Results

Held-out test set: **VCT 2025–2026, 844 matches**. The models never saw these
years during training or tuning.

| Model | Accuracy | ROC-AUC | F1 | Log-Loss | Brier |
| --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.585 | **0.625** | 0.595 | 1.299 | 0.332 |
| Random Forest | **0.592** | 0.625 | **0.595** | **0.715** | **0.256** |
| XGBoost | 0.589 | 0.617 | 0.586 | 0.721 | 0.258 |
| *Ensemble (average)* | 0.589 | 0.621 | 0.589 | 0.759 | 0.272 |

The ensemble is what the web app reports. It is not the single best model on
any one metric — averaging trades a little peak performance for stability
across the three, which is the intended behaviour.

**How to read these numbers.** A majority-class baseline (always predict Team
A) scores 0.512 accuracy on this test set, so accuracy alone is a weak signal.
ROC-AUC of ~0.62 is the honest headline: real predictive signal, well short of
a solved problem. Random Forest's Brier score of 0.256 means its probabilities
are the best calibrated — Logistic Regression's log-loss of 1.299 shows it
producing confident predictions that are frequently wrong.

Match outcomes in esports carry substantial irreducible randomness. Roster
changes, patch metas, map picks and bans, and stand-ins are all invisible to
this feature set. A ceiling well below 0.70 accuracy is expected.

The plots backing these numbers are in `ml/plots/` and rendered on `/analysis/`.

---

## Method

### 1. Rolling player history (the leakage guarantee)

For every player-map row, the `hist_*` columns hold that player's expanding
average across **all previous maps only** — `expanding().mean().shift(1)`.
Career debuts, which have no history, are filled with the global mean.

This is the pipeline's central correctness claim, and it is what
`esports/tests.py::RollingHistoryTests` exists to protect. Rows are sorted by
year → Match ID → Game ID before the expanding window runs, so history follows
chronology rather than the order rows happen to sit in the CSV.

### 2. Team aggregation

Player rows are collapsed to one row per (match, team), using only each
player's **first map appearance** in that match so the aggregate stays strictly
pre-match. Eleven team-level statistics come out: means for ACS, KD, KAST, ADR,
first kills, first deaths, headshot % and rating, plus standard deviation of
ACS (roster consistency), maximum ACS (peak performer) and mean maps played
(experience).

### 3. Differential features

Each match becomes a single row of Team A − Team B differences. A positive
value favours Team A.

| Feature | Source | Direction |
| --- | --- | --- |
| `delta_ACS` | Average Combat Score | higher is better |
| `delta_KD` | Kill/Death ratio | higher is better |
| `delta_KAST` | Kill/Assist/Trade/Survive % | higher is better |
| `delta_ADR` | Average Damage Per Round | higher is better |
| `delta_FK` | First kills per map | higher is better |
| `delta_FD` | First deaths per map | lower is better |
| `delta_HS` | Headshot % | higher is better |
| `delta_rating` | VLR composite rating | higher is better |
| `delta_consistency` | Std dev of team ACS | lower is more consistent |
| `delta_peak` | Top fragger's ACS | higher is better |
| `delta_experience` | Mean maps played | higher is better |

### 4. Time-based split

| Split | Years | Matches |
| --- | --- | --- |
| Train | 2021–2023 | 11,280 |
| Validation | 2024 | 434 |
| Test | 2025–2026 | 844 |

Splitting by time rather than randomly is what keeps the evaluation honest: a
random split would let a 2026 match inform predictions about a 2022 one.

### 5. Models

Tuned with 5-fold cross-validation on the training set, scoring ROC-AUC.

- **Logistic Regression** — `C=0.1`, standardised inputs
- **Random Forest** — 300 trees, `max_depth=8`, `min_samples_leaf=15`, `max_features='log2'`
- **XGBoost** — randomised search over depth, learning rate, subsampling; `scale_pos_weight` set to the train-set class ratio

The final prediction is the unweighted mean of the three probabilities.

---

## Retraining

Retraining needs the raw dataset — **"Valorant Champion Tour 2021–2026 Data"**
by Ryan Luong on Kaggle — which is **not** included in this repository. See
[DATA.md](DATA.md) for the exact source, download steps, and expected file
layout. Point `VALORANT_DATA_ROOT` at your unzipped copy:

```bash
set VALORANT_DATA_ROOT=C:\path\to\Valorantdata     # Windows
# export VALORANT_DATA_ROOT=/path/to/Valorantdata  # macOS / Linux

python -m ml.train
```

Expected layout:

```
Valorantdata/
├── all_ids/
│   └── all_matches_games_ids.csv
├── vct_2021/
│   └── matches/
│       ├── overview.csv
│       └── scores.csv
├── vct_2022/ …
└── vct_2026/
```

`ml/train.py` rewrites everything in `ml/saved_models/` and `ml/plots/`.
**Restart the Django server afterwards** — models and analysis metrics are
loaded once at startup and cached for the life of the process.

---

## Tests

```bash
python manage.py test esports
```

24 tests covering the no-leakage guarantee in feature engineering, the
differential feature construction, all three pages, and the prediction API's
validation and response shape. Tests that need trained artefacts skip
themselves cleanly when `ml/saved_models/` is empty.

---

## Layout

```
manage.py                  Django entry point
requirements.txt           Full app dependencies
prediction_model/          Django project settings and root URLs
esports/                   Web app
├── views.py               Page views + /api/predict/ JSON endpoint
├── predictor.py           Loads models once at startup; serving-time inference
├── tests.py               Test suite
├── templates/esports/     home / analysis / teams
└── static/esports/        style.css, predict.js
ml/                        ML pipeline (importable, framework-independent)
├── config.py              Paths, year splits, feature definitions
├── data_loader.py         CSV loading and cleaning
├── features.py            Rolling history and differential features
├── train.py               Full training pipeline
├── predict.py             Command-line prediction
├── saved_models/          Trained artefacts (committed)
└── plots/                 Diagnostic figures (committed)
```

`ml/` has no Django dependency and can be used on its own.

---

## Notes and limitations

- **Team identity is fuzzy.** Team names come straight from VLR.gg, so the
  dropdowns contain 3,562 distinct strings including qualifier and amateur
  teams with almost no history — only 259 have 20 or more matches, and 1,415
  appear in a single match. Predictions for teams with few recorded matches
  lean heavily on global-mean fill-ins and should not be trusted. The UI does
  not currently distinguish them, so pick established orgs when demoing.
- **Rosters are the latest known.** Serving-time predictions use each team's
  most recent roster in the dataset, which may be stale relative to a real
  upcoming match.
- **No match context.** Map pool, patch version, bracket stage, LAN vs online
  and roster changes are not modelled.
- `DEBUG = True` and the committed `SECRET_KEY` are fine for local thesis use
  but must be changed before any public deployment.
