"""
Loads trained ML models once at Django startup.
All other modules import from here — never re-load models per request.
"""
import os
from functools import lru_cache

import numpy as np
import pandas as pd
import joblib
from xgboost import XGBClassifier

from ml.config import MODELS_DIR, PLAYER_STAT_COLS, FEATURE_COLS, TEST_YEARS, TRAIN_YEARS, VAL_YEARS
from ml.features import DELTA_SPEC

# Human-readable metadata for each differential feature
FEATURE_META = [
    {'key': 'delta_ACS',         'label': 'Combat Score (ACS)',          'col': 'avg_ACS',    'higher_better': True,  'fmt': 'float1'},
    {'key': 'delta_KD',          'label': 'Kill / Death Ratio',           'col': 'avg_KD',     'higher_better': True,  'fmt': 'float2'},
    {'key': 'delta_KAST',        'label': 'KAST %',                      'col': 'avg_KAST',   'higher_better': True,  'fmt': 'pct'},
    {'key': 'delta_ADR',         'label': 'Avg Damage Per Round',         'col': 'avg_ADR',    'higher_better': True,  'fmt': 'float1'},
    {'key': 'delta_FK',          'label': 'First Kills Per Map',          'col': 'avg_FK',     'higher_better': True,  'fmt': 'float1'},
    {'key': 'delta_FD',          'label': 'First Deaths Per Map',         'col': 'avg_FD',     'higher_better': False, 'fmt': 'float1'},
    {'key': 'delta_HS',          'label': 'Headshot %',                   'col': 'avg_HS',     'higher_better': True,  'fmt': 'pct'},
    {'key': 'delta_rating',      'label': 'VLR Rating',                   'col': 'avg_rating', 'higher_better': True,  'fmt': 'float2'},
    {'key': 'delta_consistency', 'label': 'Roster Consistency (std ACS)', 'col': 'std_ACS',    'higher_better': False, 'fmt': 'float1'},
    {'key': 'delta_peak',        'label': 'Peak Performer ACS',           'col': 'peak_ACS',   'higher_better': True,  'fmt': 'float1'},
    {'key': 'delta_experience',  'label': 'Experience (Maps Played)',     'col': 'avg_exp',    'higher_better': True,  'fmt': 'int'},
]


def _load_models():
    lr      = joblib.load(os.path.join(MODELS_DIR, 'lr.pkl'))
    scaler  = joblib.load(os.path.join(MODELS_DIR, 'scaler.pkl'))
    rf      = joblib.load(os.path.join(MODELS_DIR, 'rf.pkl'))
    xgb     = XGBClassifier()
    xgb.load_model(os.path.join(MODELS_DIR, 'xgb.json'))
    players = pd.read_parquet(os.path.join(MODELS_DIR, 'player_latest_stats.parquet'))
    roster  = pd.read_parquet(os.path.join(MODELS_DIR, 'team_latest_roster.parquet'))
    return lr, scaler, rf, xgb, players, roster


try:
    _lr, _scaler, _rf, _xgb, _player_stats, _team_roster = _load_models()
    _ready = True
    _error = None
except Exception as exc:
    _ready = False
    _error = str(exc)


def models_ready():
    return _ready


def get_all_teams():
    if not _ready:
        return []
    return sorted(_team_roster['Team'].dropna().unique().tolist())


def _team_features(team_name: str) -> dict:
    players = _team_roster[_team_roster['Team'] == team_name]['Player'].tolist()
    if not players:
        mask = _team_roster['Team'].str.lower().str.contains(
            team_name.lower(), regex=False, na=False
        )
        players = _team_roster[mask]['Player'].tolist()
    if not players:
        raise ValueError(f"Team '{team_name}' not found. Check the spelling on the Teams page.")

    p = _player_stats[_player_stats['Player'].isin(players)]
    if p.empty:
        raise ValueError(f"No historical stats found for '{team_name}'.")

    acs = 'hist_Average Combat Score'
    return {
        'avg_ACS':    float(p[acs].mean()),
        'std_ACS':    float(p[acs].std(ddof=0)) if len(p) > 1 else 0.0,
        'peak_ACS':   float(p[acs].max()),
        'avg_KD':     float(p['hist_Kills - Deaths (KD)'].mean()),
        'avg_KAST':   float(p['hist_KAST_pct'].mean()),
        'avg_ADR':    float(p['hist_Average Damage Per Round'].mean()),
        'avg_FK':     float(p['hist_First Kills'].mean()),
        'avg_FD':     float(p['hist_First Deaths'].mean()),
        'avg_HS':     float(p['hist_HS_pct'].mean()),
        'avg_rating': float(p['hist_Rating'].mean()),
        'avg_exp':    float(p['hist_maps_played'].mean()),
    }


def predict_match(team_a: str, team_b: str) -> dict:
    if not _ready:
        raise RuntimeError(f"Models not loaded: {_error}. Run python -m ml.train first.")

    fa = _team_features(team_a)
    fb = _team_features(team_b)

    # Order must match the training matrix exactly, so drive it off FEATURE_COLS
    feat_vec = np.array([[fa[DELTA_SPEC[f]] - fb[DELTA_SPEC[f]] for f in FEATURE_COLS]])

    lr_prob  = float(_lr.predict_proba(_scaler.transform(feat_vec))[0, 1])
    rf_prob  = float(_rf.predict_proba(feat_vec)[0, 1])
    xgb_prob = float(_xgb.predict_proba(feat_vec)[0, 1])
    ensemble = (lr_prob + rf_prob + xgb_prob) / 3.0

    # Per-feature breakdown for the UI
    features = []
    for meta in FEATURE_META:
        col  = meta['col']
        a_v  = fa[col]
        b_v  = fb[col]
        hi   = meta['higher_better']
        a_wins = (hi and a_v > b_v) or (not hi and a_v < b_v)
        total  = a_v + b_v
        if total > 0:
            a_prop = (a_v / total) if hi else (b_v / total)
        else:
            a_prop = 0.5
        a_prop = max(0.05, min(0.95, a_prop))
        features.append({
            'key':         meta['key'],
            'label':       meta['label'],
            'fmt':         meta['fmt'],
            'a_val':       a_v,
            'b_val':       b_v,
            'a_proportion': round(a_prop * 100, 1),
            'b_proportion': round((1 - a_prop) * 100, 1),
            'a_wins':      a_wins,
        })

    model_winners = [
        team_a if lr_prob  >= 0.5 else team_b,
        team_a if rf_prob  >= 0.5 else team_b,
        team_a if xgb_prob >= 0.5 else team_b,
    ]

    return {
        'team_a':      team_a,
        'team_b':      team_b,
        'team_a_prob': round(ensemble * 100, 1),
        'team_b_prob': round((1 - ensemble) * 100, 1),
        'lr_prob_a':   round(lr_prob  * 100, 1),
        'rf_prob_a':   round(rf_prob  * 100, 1),
        'xgb_prob_a':  round(xgb_prob * 100, 1),
        'winner':      team_a if ensemble >= 0.5 else team_b,
        'confidence':  round(max(ensemble, 1 - ensemble) * 100, 1),
        'all_agree':   len(set(model_winners)) == 1,
        'features':    features,
    }


# Metric key → True if a higher value is better. Drives the "best" highlight.
_METRIC_DIRECTION = {
    'accuracy': True,
    'auc':      True,
    'f1':       True,
    'logloss':  False,
    'brier':    False,
}


@lru_cache(maxsize=1)
def get_analysis_metrics() -> dict:
    """
    Test-set metrics for the analysis page.

    Cached: the artefacts are immutable for the life of the process, so this
    only ever runs once. Restart the server after retraining.
    """
    from sklearn.metrics import (
        accuracy_score, roc_auc_score, f1_score, log_loss, brier_score_loss
    )

    feature_df = pd.read_parquet(os.path.join(MODELS_DIR, 'feature_df.parquet'))
    train_df   = feature_df[feature_df['_year'].isin(TRAIN_YEARS)]
    val_df     = feature_df[feature_df['_year'].isin(VAL_YEARS)]
    test_df    = feature_df[feature_df['_year'].isin(TEST_YEARS)]

    X_test    = test_df[FEATURE_COLS].values
    y_test    = test_df['label'].values
    X_test_sc = _scaler.transform(X_test)

    probs = {}
    rows  = []
    for name, model, X in [
        ('Logistic Regression', _lr,  X_test_sc),
        ('Random Forest',       _rf,  X_test),
        ('XGBoost',             _xgb, X_test),
    ]:
        probs[name] = model.predict_proba(X)[:, 1]

    # The site predicts with the mean of the three, so evaluate that too —
    # otherwise the published metrics describe models the UI never uses alone.
    probs['Ensemble (average)'] = np.mean(list(probs.values()), axis=0)

    for name, y_prob in probs.items():
        y_pred = (y_prob >= 0.5).astype(int)
        rows.append({
            'name':      name,
            'ensemble':  name.startswith('Ensemble'),
            'accuracy':  accuracy_score(y_test, y_pred),
            'auc':       roc_auc_score(y_test, y_prob),
            'f1':        f1_score(y_test, y_pred),
            'logloss':   log_loss(y_test, y_prob),
            'brier':     brier_score_loss(y_test, y_prob),
        })

    # Flag the winning model per metric instead of hardcoding the numbers.
    for key, higher_better in _METRIC_DIRECTION.items():
        best = (max if higher_better else min)(r[key] for r in rows)
        for r in rows:
            r[f'{key}_best'] = r[key] == best

    # Format for display only after the comparison is done.
    for r in rows:
        for key in _METRIC_DIRECTION:
            r[key] = f"{r[key]:.3f}"

    baseline = float(test_df['label'].mean())

    return {
        'metrics':      rows,
        'train_count':  len(train_df),
        'val_count':    len(val_df),
        'test_count':   len(test_df),
        'total_count':  len(feature_df),
        'feature_count': len(FEATURE_COLS),
        'model_count':  3,
        # Always-pick-Team-A accuracy: the floor any model has to beat.
        'baseline_acc': f"{max(baseline, 1 - baseline):.3f}",
    }
