"""
Predict the win probability for an upcoming match between two teams.

Usage:
    python -m ml.predict --team_a "Sentinels" --team_b "T1"
    python -m ml.predict --team_a "Paper Rex" --team_b "Gen.G"

Players are selected from each team's most recent known roster in the dataset.
Run ml/train.py first to generate the saved models and artefacts.
"""
import argparse
import os
import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from ml.config import MODELS_DIR, FEATURE_COLS
from ml.features import DELTA_SPEC


def _load_artifacts():
    missing = [p for p in ['lr.pkl', 'scaler.pkl', 'rf.pkl', 'xgb.json',
                            'player_latest_stats.parquet',
                            'team_latest_roster.parquet']
               if not os.path.exists(os.path.join(MODELS_DIR, p))]
    if missing:
        raise FileNotFoundError(
            f"Missing artefacts: {missing}\n"
            "Run  python -m ml.train  first."
        )

    lr     = joblib.load(os.path.join(MODELS_DIR, 'lr.pkl'))
    scaler = joblib.load(os.path.join(MODELS_DIR, 'scaler.pkl'))
    rf     = joblib.load(os.path.join(MODELS_DIR, 'rf.pkl'))
    xgb    = XGBClassifier()
    xgb.load_model(os.path.join(MODELS_DIR, 'xgb.json'))

    player_stats = pd.read_parquet(os.path.join(MODELS_DIR, 'player_latest_stats.parquet'))
    team_roster  = pd.read_parquet(os.path.join(MODELS_DIR, 'team_latest_roster.parquet'))
    return lr, scaler, rf, xgb, player_stats, team_roster


def _team_features(team_name, player_stats, team_roster):
    players = team_roster[team_roster['Team'] == team_name]['Player'].tolist()
    if not players:
        # Fuzzy fallback: case-insensitive partial match
        mask = team_roster['Team'].str.lower().str.contains(
            team_name.lower(), regex=False, na=False
        )
        players = team_roster[mask]['Player'].tolist()
        if not players:
            all_teams = sorted(team_roster['Team'].dropna().unique())
            raise ValueError(
                f"Team '{team_name}' not found.\n"
                f"Available teams (sample): {all_teams[:30]}"
            )

    p_data = player_stats[player_stats['Player'].isin(players)]
    if p_data.empty:
        raise ValueError(f"No stat history for any player on '{team_name}'.")

    acs_col = 'hist_Average Combat Score'
    return {
        'avg_ACS':    p_data[acs_col].mean(),
        'std_ACS':    p_data[acs_col].std(ddof=0),
        'peak_ACS':   p_data[acs_col].max(),
        'avg_KD':     p_data['hist_Kills - Deaths (KD)'].mean(),
        'avg_KAST':   p_data['hist_KAST_pct'].mean(),
        'avg_ADR':    p_data['hist_Average Damage Per Round'].mean(),
        'avg_FK':     p_data['hist_First Kills'].mean(),
        'avg_FD':     p_data['hist_First Deaths'].mean(),
        'avg_HS':     p_data['hist_HS_pct'].mean(),
        'avg_rating': p_data['hist_Rating'].mean(),
        'avg_exp':    p_data['hist_maps_played'].mean(),
    }


def predict_match(team_a: str, team_b: str) -> dict:
    lr, scaler, rf, xgb, player_stats, team_roster = _load_artifacts()

    fa = _team_features(team_a, player_stats, team_roster)
    fb = _team_features(team_b, player_stats, team_roster)

    # Build feature vector in exactly the order the models were trained on
    feat_vec = np.array([[fa[DELTA_SPEC[f]] - fb[DELTA_SPEC[f]] for f in FEATURE_COLS]])

    lr_prob  = lr.predict_proba(scaler.transform(feat_vec))[0, 1]
    rf_prob  = rf.predict_proba(feat_vec)[0, 1]
    xgb_prob = xgb.predict_proba(feat_vec)[0, 1]
    ensemble = np.mean([lr_prob, rf_prob, xgb_prob])

    print(f"\n{'='*52}")
    print(f"  {team_a}  vs  {team_b}")
    print(f"{'='*52}")
    print(f"{'Model':<26}  {team_a[:14]:>14}  {team_b[:14]:>14}")
    print(f"{'-'*52}")
    for name, p in [('Logistic Regression', lr_prob),
                    ('Random Forest',       rf_prob),
                    ('XGBoost',             xgb_prob),
                    ('--- Ensemble avg ---', ensemble)]:
        print(f"  {name:<24}  {p*100:>6.1f}%      {(1-p)*100:>6.1f}%")
    print(f"{'='*52}")
    winner     = team_a if ensemble >= 0.5 else team_b
    confidence = max(ensemble, 1 - ensemble)
    print(f"  Predicted winner: {winner}  ({confidence*100:.1f}% confidence)")
    print()

    return {
        'team_a': team_a,
        'team_b': team_b,
        'lr_prob':  lr_prob,
        'rf_prob':  rf_prob,
        'xgb_prob': xgb_prob,
        'ensemble': ensemble,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict a Valorant match outcome')
    parser.add_argument('--team_a', required=True)
    parser.add_argument('--team_b', required=True)
    args = parser.parse_args()
    predict_match(args.team_a, args.team_b)
