import os
import pandas as pd
from ml.config import DATA_ROOT, YEARS


def load_overview(years=None):
    """
    Load overview.csv from all years, keep only side='both' rows,
    and parse KAST% / HS% from strings to floats.
    """
    if years is None:
        years = YEARS
    dfs = []
    for yr in years:
        path = os.path.join(DATA_ROOT, yr, 'matches', 'overview.csv')
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path, low_memory=False)
        df['_year'] = yr
        dfs.append(df)

    if not dfs:
        raise FileNotFoundError(f"No overview.csv files found under {DATA_ROOT}")

    combined = pd.concat(dfs, ignore_index=True)
    combined = combined[combined['Side'] == 'both'].copy()

    combined['KAST_pct'] = (
        pd.to_numeric(combined['Kill, Assist, Trade, Survive %'].str.rstrip('%'), errors='coerce') / 100
    )
    combined['HS_pct'] = (
        pd.to_numeric(combined['Headshot %'].str.rstrip('%'), errors='coerce') / 100
    )

    for col in ['Rating', 'Average Combat Score', 'Kills - Deaths (KD)',
                'Average Damage Per Round', 'First Kills', 'First Deaths']:
        combined[col] = pd.to_numeric(combined[col], errors='coerce')

    combined = combined.dropna(subset=['Player'])
    combined['Player'] = combined['Player'].str.strip()

    keep = [
        '_year', 'Tournament', 'Stage', 'Match Type', 'Match Name', 'Map',
        'Player', 'Team',
        'Rating', 'Average Combat Score', 'Kills - Deaths (KD)',
        'KAST_pct', 'Average Damage Per Round',
        'First Kills', 'First Deaths', 'HS_pct',
    ]
    return combined[keep].copy()


def load_scores(years=None):
    """
    Load scores.csv, drop draws and TBD placeholder teams,
    and create a binary label: 1 = Team A won.
    """
    if years is None:
        years = YEARS
    dfs = []
    for yr in years:
        path = os.path.join(DATA_ROOT, yr, 'matches', 'scores.csv')
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        df['_year'] = yr
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    combined = combined[combined['Match Result'] != 'Draw']
    combined = combined[~combined['Team A'].str.upper().str.contains('TBD', na=False)]
    combined = combined[~combined['Team B'].str.upper().str.contains('TBD', na=False)]

    combined['label'] = combined.apply(
        lambda r: 1 if r['Match Result'].strip() == r['Team A'].strip() + ' won' else 0,
        axis=1
    )

    keep = ['_year', 'Tournament', 'Stage', 'Match Type', 'Match Name',
            'Team A', 'Team B', 'Team A Score', 'Team B Score', 'label']
    return combined[keep].copy()


def load_match_ids():
    """
    Load cross-year match/game ID table.
    Returns one row per map per match (needed for Game ID sort within a match).
    """
    path = os.path.join(DATA_ROOT, 'all_ids', 'all_matches_games_ids.csv')
    df = pd.read_csv(path)
    return df[['Tournament', 'Stage', 'Match Type', 'Match Name',
               'Map', 'Match ID', 'Game ID']].copy()
