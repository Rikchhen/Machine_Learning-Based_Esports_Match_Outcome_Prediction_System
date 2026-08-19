# Dataset

## Source

**"Valorant Champion Tour 2021–2026 Data"** by **Ryan Luong**, on Kaggle:

<https://www.kaggle.com/datasets/ryanluong1/valorant-champion-tour-2021-2023-data>

The URL slug still reads `2021-2023` (its original name), but the dataset has
been extended through 2026 — which is why `ml/config.py` lists `vct_2021`
through `vct_2026`.

This is **secondary, publicly available, quantitative** match/player data
scraped from VLR.gg. No human participants, no personal data.

## Do I need to re-download it?

**Not to run the app.** The trained artefacts in `ml/saved_models/` are
committed, so the Django site and predictions work without the raw CSVs.

You only need the raw dataset to **re-run training** (`python -m ml.train`) or
to verify the pipeline end to end.

## Restore steps

1. Download from the Kaggle page above (free account required), or via the CLI:
   ```bash
   kaggle datasets download -d ryanluong1/valorant-champion-tour-2021-2023-data
   ```
2. Unzip so the folders sit at the path `ml/config.py` expects
   (`C:\Users\ACER\Downloads\Valorantdata`, or wherever `VALORANT_DATA_ROOT`
   points):
   ```
   Valorantdata/
   ├── all_ids/
   │   └── all_matches_games_ids.csv
   ├── vct_2021/
   │   └── matches/
   │       ├── overview.csv
   │       └── scores.csv
   ├── vct_2022/ …
   └── … through vct_2026/
   ```
3. Sanity check: `python -m ml.train` should reproduce **12,558 matches**
   (7,107 / 3,842 / 331 / 434 / 502 / 342 rows for 2021–2026 respectively).

To use a different location instead of `Downloads\Valorantdata`, set the
environment variable `VALORANT_DATA_ROOT` to point at your unzipped folder.

## Files and columns the pipeline uses

Only these files are read (`ml/data_loader.py`); the dataset contains more.

| File | Columns used |
| --- | --- |
| `<year>/matches/overview.csv` | `Side` (kept where `= 'both'`), `Tournament`, `Stage`, `Match Type`, `Match Name`, `Map`, `Player`, `Team`, `Rating`, `Average Combat Score`, `Kills - Deaths (KD)`, `Kill, Assist, Trade, Survive %`, `Average Damage Per Round`, `First Kills`, `First Deaths`, `Headshot %` |
| `<year>/matches/scores.csv` | `Tournament`, `Stage`, `Match Type`, `Match Name`, `Team A`, `Team B`, `Team A Score`, `Team B Score`, `Match Result` |
| `all_ids/all_matches_games_ids.csv` | `Tournament`, `Stage`, `Match Type`, `Match Name`, `Map`, `Match ID`, `Game ID` |

If a future dataset version renames these columns, the loader will raise a
`KeyError` — check the schema against this table first.

## Citing it (thesis)

Kaggle datasets get updated, so the **version and access date** are part of
reproducibility. A CU-Harvard-style reference looks roughly like:

> Luong, R. (2026) *Valorant Champion Tour 2021–2026 Data*. Kaggle. Available
> at: <https://www.kaggle.com/datasets/ryanluong1/valorant-champion-tour-2021-2023-data>
> (Accessed: [date]).

**Confirm the author name spelling, year, and "last updated" version on the
Kaggle page itself before putting this in your references** — don't take the
formatting above on trust for something that gets marked.
