import pandas as pd
from ml.config import PLAYER_STAT_COLS, TEAM_FEAT_COLS, FEATURE_COLS

JOIN_KEYS = ['Tournament', 'Stage', 'Match Type', 'Match Name']

# Maps each delta feature name → the team-level column it's computed from
DELTA_SPEC = {
    'delta_ACS':         'avg_ACS',
    'delta_KD':          'avg_KD',
    'delta_KAST':        'avg_KAST',
    'delta_ADR':         'avg_ADR',
    'delta_FK':          'avg_FK',
    'delta_FD':          'avg_FD',
    'delta_HS':          'avg_HS',
    'delta_rating':      'avg_rating',
    'delta_consistency': 'std_ACS',
    'delta_peak':        'peak_ACS',
    'delta_experience':  'avg_exp',
}

# The model input vector is always built by iterating FEATURE_COLS, so the two
# must stay in sync. Fail loudly at import time rather than silently feeding a
# permuted vector to the models.
assert set(DELTA_SPEC) == set(FEATURE_COLS), (
    f"DELTA_SPEC and FEATURE_COLS disagree: "
    f"{set(DELTA_SPEC) ^ set(FEATURE_COLS)}"
)


def build_rolling_player_history(overview_df, match_ids_df):
    """
    Attach expanding-window historical averages to every player-map row.

    For each row, the hist_* columns represent that player's average stats
    across ALL PREVIOUS maps they have played (shifted by 1, so the current
    map is excluded). This is the only information available before a match
    and therefore has zero data leakage.

    Rows with no prior history (career debut) are filled with the global mean.
    """
    df = overview_df.merge(
        match_ids_df[JOIN_KEYS + ['Map', 'Match ID', 'Game ID']],
        on=JOIN_KEYS + ['Map'],
        how='left',
    )

    df['_year_num'] = df['_year'].str.extract(r'(\d+)').astype(int)
    # Sort: year first, then match within year, then map within match
    df = df.sort_values(
        ['_year_num', 'Match ID', 'Game ID'], na_position='last'
    ).reset_index(drop=True)

    global_means = df[PLAYER_STAT_COLS].mean()

    for col in PLAYER_STAT_COLS:
        series = df.groupby('Player')[col].transform(
            lambda x: x.expanding().mean().shift(1)
        )
        df[f'hist_{col}'] = series.fillna(global_means[col])

    # Number of maps played before this one (experience indicator; 0 = debut)
    df['hist_maps_played'] = df.groupby('Player').cumcount()

    return df


def _aggregate_team_stats(rolling_df):
    """
    Collapse rolling_df to one row per (match, team).

    Uses only the FIRST map appearance of each player per match so that
    hist_* values reflect strictly pre-match history.
    """
    hist_cols = [f'hist_{c}' for c in PLAYER_STAT_COLS]

    player_per_match = (
        rolling_df
        .drop_duplicates(subset=JOIN_KEYS + ['Player'], keep='first')
        [JOIN_KEYS + ['Team', 'hist_maps_played'] + hist_cols]
    )

    return player_per_match.groupby(JOIN_KEYS + ['Team']).agg(
        avg_ACS  =('hist_Average Combat Score',    'mean'),
        std_ACS  =('hist_Average Combat Score',    lambda x: x.std(ddof=0)),
        peak_ACS =('hist_Average Combat Score',    'max'),
        avg_KD   =('hist_Kills - Deaths (KD)',      'mean'),
        avg_KAST =('hist_KAST_pct',                'mean'),
        avg_ADR  =('hist_Average Damage Per Round', 'mean'),
        avg_FK   =('hist_First Kills',              'mean'),
        avg_FD   =('hist_First Deaths',             'mean'),
        avg_HS   =('hist_HS_pct',                  'mean'),
        avg_rating=('hist_Rating',                 'mean'),
        avg_exp  =('hist_maps_played',              'mean'),
    ).reset_index()


def build_match_features(scores_df, rolling_df):
    """
    Build one row per match with 11 differential (Team A − Team B) features.

    Returns a DataFrame with JOIN_KEYS + [_year, Team A, Team B, label] + FEATURE_COLS.
    Matches where either team's players cannot be found in overview are dropped.
    """
    team_stats = _aggregate_team_stats(rolling_df)

    # Merge Team A stats
    a_stats = team_stats.rename(columns={c: f'a_{c}' for c in TEAM_FEAT_COLS})
    df = scores_df.merge(
        a_stats,
        left_on=JOIN_KEYS + ['Team A'],
        right_on=JOIN_KEYS + ['Team'],
        how='inner',
    ).drop(columns=['Team'])

    # Merge Team B stats
    b_stats = team_stats.rename(columns={c: f'b_{c}' for c in TEAM_FEAT_COLS})
    df = df.merge(
        b_stats,
        left_on=JOIN_KEYS + ['Team B'],
        right_on=JOIN_KEYS + ['Team'],
        how='inner',
    ).drop(columns=['Team'])

    # Differential features: positive value favours Team A
    for feat, col in DELTA_SPEC.items():
        df[feat] = df[f'a_{col}'] - df[f'b_{col}']

    out_cols = JOIN_KEYS + ['_year', 'Team A', 'Team B', 'label'] + FEATURE_COLS
    return df[out_cols].dropna(subset=FEATURE_COLS).reset_index(drop=True)
