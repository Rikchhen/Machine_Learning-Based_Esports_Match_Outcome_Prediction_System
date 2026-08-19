import os

# Raw VLR.gg CSV dump. Only needed to re-run training — serving the web app
# uses the artefacts in saved_models/ and never touches DATA_ROOT.
# Override with the VALORANT_DATA_ROOT environment variable.
DATA_ROOT = os.environ.get(
    'VALORANT_DATA_ROOT',
    r'C:\Users\ACER\Downloads\Valorantdata',
)
ML_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(ML_DIR, 'saved_models')
PLOTS_DIR = os.path.join(ML_DIR, 'plots')

YEARS = ['vct_2021', 'vct_2022', 'vct_2023', 'vct_2024', 'vct_2025', 'vct_2026']
TRAIN_YEARS = {'vct_2021', 'vct_2022', 'vct_2023'}
VAL_YEARS   = {'vct_2024'}
TEST_YEARS  = {'vct_2025', 'vct_2026'}

# Stat columns from overview.csv (after parsing % strings)
PLAYER_STAT_COLS = [
    'Average Combat Score',
    'Kills - Deaths (KD)',
    'KAST_pct',
    'Average Damage Per Round',
    'First Kills',
    'First Deaths',
    'HS_pct',
    'Rating',
]

# Per-team aggregated features produced by _aggregate_team_stats()
TEAM_FEAT_COLS = [
    'avg_ACS', 'std_ACS', 'peak_ACS',
    'avg_KD', 'avg_KAST', 'avg_ADR',
    'avg_FK', 'avg_FD', 'avg_HS',
    'avg_rating', 'avg_exp',
]

# Final model input features (Team A - Team B differentials)
FEATURE_COLS = [
    'delta_ACS', 'delta_KD', 'delta_KAST', 'delta_ADR',
    'delta_FK', 'delta_FD', 'delta_HS', 'delta_rating',
    'delta_consistency', 'delta_peak', 'delta_experience',
]
