"""
Full training pipeline for the Valorant match prediction model.

Run from the project root:
    python -m ml.train
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    log_loss, brier_score_loss, roc_curve, classification_report,
)
from xgboost import XGBClassifier

import shap

from ml.config import (
    TRAIN_YEARS, VAL_YEARS, TEST_YEARS,
    PLAYER_STAT_COLS, FEATURE_COLS,
    MODELS_DIR, PLOTS_DIR,
)
from ml.data_loader import load_overview, load_scores, load_match_ids
from ml.features import build_rolling_player_history, build_match_features


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(model, X, y, name):
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]
    print(f"\n{'-'*50}")
    print(f"  {name}")
    print(f"{'-'*50}")
    print(f"  Accuracy  : {accuracy_score(y, y_pred):.4f}")
    print(f"  ROC-AUC   : {roc_auc_score(y, y_prob):.4f}")
    print(f"  F1        : {f1_score(y, y_pred):.4f}")
    print(f"  Log-loss  : {log_loss(y, y_prob):.4f}")
    print(f"  Brier     : {brier_score_loss(y, y_prob):.4f}")
    print(classification_report(y, y_pred,
                                target_names=['Team B wins', 'Team A wins'],
                                zero_division=0))
    return y_prob


# ---------------------------------------------------------------------------
# Hyperparameter tuning
# ---------------------------------------------------------------------------

def tune_lr(X_train, y_train):
    print("  Tuning Logistic Regression...")
    gs = GridSearchCV(
        LogisticRegression(max_iter=2000, random_state=42),
        {'C': [0.001, 0.01, 0.05, 0.1, 0.5, 1.0]},
        cv=5, scoring='roc_auc', n_jobs=-1,
    )
    gs.fit(X_train, y_train)
    print(f"    Best C={gs.best_params_['C']}  CV AUC={gs.best_score_:.4f}")
    return gs.best_estimator_


def tune_rf(X_train, y_train):
    print("  Tuning Random Forest...")
    rs = RandomizedSearchCV(
        RandomForestClassifier(random_state=42, n_jobs=-1),
        {
            'n_estimators':    [100, 200, 300, 400],
            'max_depth':       [4, 6, 8, None],
            'min_samples_leaf':[2, 5, 10, 15],
            'max_features':    ['sqrt', 'log2'],
        },
        n_iter=30, cv=5, scoring='roc_auc',
        random_state=42, n_jobs=1,   # n_jobs=1 avoids nested parallelism with RF
    )
    rs.fit(X_train, y_train)
    print(f"    Best params: {rs.best_params_}  CV AUC={rs.best_score_:.4f}")
    return rs.best_estimator_


def tune_xgb(X_train, y_train, scale_pos_weight):
    print(f"  Tuning XGBoost (scale_pos_weight={scale_pos_weight:.3f})...")
    rs = RandomizedSearchCV(
        XGBClassifier(
            n_estimators=400,
            scale_pos_weight=scale_pos_weight,
            eval_metric='logloss',
            verbosity=0,
            random_state=42,
        ),
        {
            'max_depth':       [3, 4, 5, 6],
            'learning_rate':   [0.01, 0.05, 0.1, 0.15],
            'subsample':       [0.6, 0.8, 1.0],
            'colsample_bytree':[0.6, 0.8, 1.0],
            'min_child_weight':[1, 3, 5],
        },
        n_iter=20, cv=5, scoring='roc_auc',
        random_state=42, n_jobs=1,   # XGBoost handles its own threading
    )
    rs.fit(X_train, y_train)
    print(f"    Best params: {rs.best_params_}  CV AUC={rs.best_score_:.4f}")
    return rs.best_estimator_


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_roc_curves(probs_dict, y_true, save_path):
    plt.figure(figsize=(8, 6))
    for name, y_prob in probs_dict.items():
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        plt.plot(fpr, tpr, label=f'{name} (AUC={auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves -- Test Set (2025-2026)')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


def plot_calibration(probs_dict, y_true, save_path):
    plt.figure(figsize=(8, 6))
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
    for name, y_prob in probs_dict.items():
        frac_pos, mean_pred = calibration_curve(
            y_true, y_prob, n_bins=10, strategy='quantile'
        )
        plt.plot(mean_pred, frac_pos, marker='o', label=name)
    plt.xlabel('Mean Predicted Probability (Team A wins)')
    plt.ylabel('Observed Win Rate')
    plt.title('Calibration Plot -- Test Set')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


def plot_importances(model, feature_names, save_path, title):
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_[0])
    else:
        return
    idx = np.argsort(importances)
    plt.figure(figsize=(8, 5))
    plt.barh([feature_names[i] for i in idx], importances[idx], color='steelblue')
    plt.xlabel('Importance')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


def plot_shap(model, X, feature_names, save_path):
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    plt.figure()
    shap.summary_plot(shap_values, X, feature_names=feature_names,
                      plot_type='dot', show=False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR,  exist_ok=True)

    # ── 1. Load raw data ────────────────────────────────────────────────────
    print("=" * 55)
    print("STEP 1: Loading data")
    print("=" * 55)
    overview_df  = load_overview()
    scores_df    = load_scores()
    match_ids_df = load_match_ids()
    print(f"  overview rows (side=both)  : {len(overview_df):,}")
    print(f"  matches (draws/TBD dropped): {len(scores_df):,}")

    # ── 2. Rolling player history ───────────────────────────────────────────
    print("\n" + "=" * 55)
    print("STEP 2: Building rolling player history")
    print("=" * 55)
    rolling_df = build_rolling_player_history(overview_df, match_ids_df)
    print(f"  Done. {len(rolling_df):,} player-map rows processed.")

    # ── 3. Match feature matrix ─────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("STEP 3: Building match feature matrix")
    print("=" * 55)
    feature_df = build_match_features(scores_df, rolling_df)
    print(f"  Matches with full features: {len(feature_df):,}")
    print(f"  Team A win rate           : {feature_df['label'].mean():.3f}")

    # ── 4. Time-based split ─────────────────────────────────────────────────
    train_df = feature_df[feature_df['_year'].isin(TRAIN_YEARS)]
    val_df   = feature_df[feature_df['_year'].isin(VAL_YEARS)]
    test_df  = feature_df[feature_df['_year'].isin(TEST_YEARS)]
    print(f"\n  Train (2021-23): {len(train_df):,} | "
          f"Val (2024): {len(val_df):,} | "
          f"Test (2025-26): {len(test_df):,}")

    X_train, y_train = train_df[FEATURE_COLS].values, train_df['label'].values
    X_val,   y_val   = val_df[FEATURE_COLS].values,   val_df['label'].values
    X_test,  y_test  = test_df[FEATURE_COLS].values,  test_df['label'].values

    # Scale for Logistic Regression
    scaler     = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_val_sc   = scaler.transform(X_val)
    X_test_sc  = scaler.transform(X_test)

    # XGBoost class-balance weight: neg/pos so both classes carry equal total weight
    spw = float((y_train == 0).sum()) / float((y_train == 1).sum())

    # ── 5. Hyperparameter tuning ────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("STEP 4: Hyperparameter tuning (5-fold CV on train set)")
    print("=" * 55)
    lr  = tune_lr(X_train_sc, y_train)
    rf  = tune_rf(X_train,    y_train)
    xgb = tune_xgb(X_train,   y_train, spw)

    # ── 6. Evaluate ─────────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("STEP 5: Evaluation -- Validation set (2024)")
    print("=" * 55)
    evaluate(lr,  X_val_sc, y_val, "Logistic Regression")
    evaluate(rf,  X_val,    y_val, "Random Forest")
    evaluate(xgb, X_val,    y_val, "XGBoost")

    print("\n" + "=" * 55)
    print("STEP 5: Evaluation -- Test set (2025-2026)")
    print("=" * 55)
    lr_prob  = evaluate(lr,  X_test_sc, y_test, "Logistic Regression")
    rf_prob  = evaluate(rf,  X_test,    y_test, "Random Forest")
    xgb_prob = evaluate(xgb, X_test,    y_test, "XGBoost")

    # ── 7. Plots ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("STEP 6: Generating plots")
    print("=" * 55)
    probs = {
        'Logistic Regression': lr_prob,
        'Random Forest':       rf_prob,
        'XGBoost':             xgb_prob,
    }
    plot_roc_curves(
        probs, y_test,
        os.path.join(PLOTS_DIR, 'roc_curves.png'),
    )
    plot_calibration(
        probs, y_test,
        os.path.join(PLOTS_DIR, 'calibration.png'),
    )
    plot_importances(lr,  FEATURE_COLS,
                     os.path.join(PLOTS_DIR, 'lr_coef.png'),
                     'Logistic Regression -- Absolute Coefficients')
    plot_importances(rf,  FEATURE_COLS,
                     os.path.join(PLOTS_DIR, 'rf_importance.png'),
                     'Random Forest -- Feature Importance')
    plot_importances(xgb, FEATURE_COLS,
                     os.path.join(PLOTS_DIR, 'xgb_importance.png'),
                     'XGBoost -- Feature Importance')

    print("  Running SHAP on XGBoost (test set)...")
    plot_shap(xgb, X_test, FEATURE_COLS,
              os.path.join(PLOTS_DIR, 'shap_summary.png'))

    # ── 8. Save artefacts ────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("STEP 7: Saving models and artefacts")
    print("=" * 55)
    joblib.dump(lr,     os.path.join(MODELS_DIR, 'lr.pkl'))
    joblib.dump(scaler, os.path.join(MODELS_DIR, 'scaler.pkl'))
    joblib.dump(rf,     os.path.join(MODELS_DIR, 'rf.pkl'))
    xgb.save_model(     os.path.join(MODELS_DIR, 'xgb.json'))
    print(f"  Models saved to {MODELS_DIR}")

    # Player and team lookup tables for predict.py
    hist_cols = [f'hist_{c}' for c in PLAYER_STAT_COLS]
    player_latest = (
        rolling_df
        .groupby('Player')
        .last()[['Team', 'hist_maps_played'] + hist_cols]
        .reset_index()
    )
    player_latest.to_parquet(
        os.path.join(MODELS_DIR, 'player_latest_stats.parquet'), index=False
    )

    team_roster = (
        rolling_df
        .drop_duplicates(subset=['Player'], keep='last')
        [['Player', 'Team']]
    )
    team_roster.to_parquet(
        os.path.join(MODELS_DIR, 'team_latest_roster.parquet'), index=False
    )

    feature_df.to_parquet(
        os.path.join(MODELS_DIR, 'feature_df.parquet'), index=False
    )
    print(f"  Lookup tables and feature matrix saved.")

    print("\nDone. All artefacts in:", MODELS_DIR)
    print("Plots in:", PLOTS_DIR)


if __name__ == '__main__':
    main()
