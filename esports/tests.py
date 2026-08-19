"""
Test suite for the VCT match predictor.

Run with:
    python manage.py test esports

The feature-engineering tests are pure and always run. The view and prediction
tests need the artefacts in ml/saved_models/ and skip themselves if training
has not been run yet.
"""
import json
import unittest

import numpy as np
import pandas as pd
from django.test import TestCase

from ml.config import FEATURE_COLS, PLAYER_STAT_COLS
from ml.features import (
    DELTA_SPEC,
    build_rolling_player_history, build_match_features,
)
from esports import predictor


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------

def _overview_row(player, team, match, map_name, acs, year='vct_2021'):
    """One player-map row with every stat column populated."""
    row = {
        '_year':      year,
        'Tournament': 'T', 'Stage': 'S', 'Match Type': 'M',
        'Match Name': match, 'Map': map_name,
        'Player': player, 'Team': team,
    }
    # ACS is the column under test; the rest just need to be non-null.
    for col in PLAYER_STAT_COLS:
        row[col] = acs if col == 'Average Combat Score' else 1.0
    return row


def _match_ids(rows):
    """Build the match/game ID table implied by a list of overview rows."""
    seen, out = set(), []
    for r in rows:
        key = (r['Match Name'], r['Map'])
        if key in seen:
            continue
        seen.add(key)
        match_no = int(r['Match Name'].split('_')[-1])
        out.append({
            'Tournament': 'T', 'Stage': 'S', 'Match Type': 'M',
            'Match Name': r['Match Name'], 'Map': r['Map'],
            'Match ID': match_no,
            'Game ID': match_no * 10 + int(r['Map'].split('_')[-1]),
        })
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# Feature engineering — the thesis' central correctness claim
# ---------------------------------------------------------------------------

class RollingHistoryTests(TestCase):
    """
    The model's validity rests on hist_* columns containing only information
    available *before* the map being predicted. These tests pin that down.
    """

    def setUp(self):
        # One player, three maps, ACS 100 -> 200 -> 300 in chronological order.
        self.rows = [
            _overview_row('alice', 'TeamA', 'match_1', 'map_1', 100),
            _overview_row('alice', 'TeamA', 'match_2', 'map_1', 200),
            _overview_row('alice', 'TeamA', 'match_3', 'map_1', 300),
        ]
        self.overview = pd.DataFrame(self.rows)
        self.ids = _match_ids(self.rows)

    def test_history_excludes_the_current_map(self):
        out = build_rolling_player_history(self.overview, self.ids)
        hist = out.sort_values('Match ID')['hist_Average Combat Score'].tolist()

        # Row 1 has no prior history -> filled with the global mean (200).
        # Row 2 sees only map 1 (100). Row 3 sees maps 1-2 (mean 150).
        # Critically, no row sees its own ACS.
        self.assertAlmostEqual(hist[0], 200.0)
        self.assertAlmostEqual(hist[1], 100.0)
        self.assertAlmostEqual(hist[2], 150.0)

    def test_experience_counter_starts_at_zero(self):
        out = build_rolling_player_history(self.overview, self.ids)
        maps = out.sort_values('Match ID')['hist_maps_played'].tolist()
        self.assertEqual(maps, [0, 1, 2])

    def test_row_order_does_not_change_the_result(self):
        """History must follow chronology, not the order rows arrive in."""
        shuffled = self.overview.iloc[::-1].reset_index(drop=True)
        a = build_rolling_player_history(self.overview, self.ids)
        b = build_rolling_player_history(shuffled, self.ids)
        col = 'hist_Average Combat Score'
        self.assertEqual(
            a.sort_values('Match ID')[col].tolist(),
            b.sort_values('Match ID')[col].tolist(),
        )

    def test_players_do_not_share_history(self):
        rows = self.rows + [
            _overview_row('bob', 'TeamB', 'match_1', 'map_1', 999),
            _overview_row('bob', 'TeamB', 'match_2', 'map_1', 999),
        ]
        out = build_rolling_player_history(pd.DataFrame(rows), _match_ids(rows))
        alice = out[out['Player'] == 'alice'].sort_values('Match ID')
        # alice's second map still sees only her own 100, not bob's 999.
        self.assertAlmostEqual(
            alice['hist_Average Combat Score'].tolist()[1], 100.0
        )


class MatchFeatureTests(TestCase):

    def test_deltas_are_team_a_minus_team_b(self):
        rows = [
            # Prior match so both teams have history going into match_2.
            _overview_row('alice', 'TeamA', 'match_1', 'map_1', 300),
            _overview_row('bob',   'TeamB', 'match_1', 'map_1', 100),
            _overview_row('alice', 'TeamA', 'match_2', 'map_1', 0),
            _overview_row('bob',   'TeamB', 'match_2', 'map_1', 0),
        ]
        rolling = build_rolling_player_history(pd.DataFrame(rows), _match_ids(rows))
        scores = pd.DataFrame([{
            '_year': 'vct_2021',
            'Tournament': 'T', 'Stage': 'S', 'Match Type': 'M',
            'Match Name': 'match_2',
            'Team A': 'TeamA', 'Team B': 'TeamB',
            'Team A Score': 2, 'Team B Score': 0, 'label': 1,
        }])

        feats = build_match_features(scores, rolling)
        self.assertEqual(len(feats), 1)
        # Going into match_2, TeamA's history is 300 and TeamB's is 100.
        self.assertAlmostEqual(feats.iloc[0]['delta_ACS'], 200.0)

    def test_delta_spec_covers_every_model_feature(self):
        """A mismatch here would feed the models a permuted vector."""
        self.assertEqual(set(DELTA_SPEC), set(FEATURE_COLS))


# ---------------------------------------------------------------------------
# Views and prediction API
# ---------------------------------------------------------------------------

requires_models = unittest.skipUnless(
    predictor.models_ready(),
    'Trained artefacts not found - run "python -m ml.train" first.',
)


class PageTests(TestCase):

    def test_home_renders(self):
        resp = self.client.get('/')
        self.assertEqual(resp.status_code, 200)
        self.assertContains(resp, 'Predict Match')

    @requires_models
    def test_home_lists_every_team_as_an_option(self):
        resp = self.client.get('/')
        teams = predictor.get_all_teams()
        # Two <select> elements, each carrying every team plus a placeholder.
        self.assertEqual(resp.content.decode().count('<option value="'), 2 * len(teams) + 2)

    def test_teams_page_renders(self):
        self.assertEqual(self.client.get('/teams/').status_code, 200)

    def test_analysis_page_renders(self):
        self.assertEqual(self.client.get('/analysis/').status_code, 200)

    @requires_models
    def test_analysis_page_reports_the_ensemble(self):
        self.assertContains(self.client.get('/analysis/'), 'Ensemble')


class PredictApiTests(TestCase):

    URL = '/api/predict/'

    def _post(self, payload):
        return self.client.post(
            self.URL, data=json.dumps(payload), content_type='application/json'
        )

    def test_get_is_rejected(self):
        self.assertEqual(self.client.get(self.URL).status_code, 405)

    def test_malformed_json_is_rejected(self):
        resp = self.client.post(
            self.URL, data='not json', content_type='application/json'
        )
        self.assertEqual(resp.status_code, 400)
        self.assertIn('error', resp.json())

    def test_missing_team_is_rejected(self):
        self.assertEqual(self._post({'team_a': 'Sentinels'}).status_code, 400)

    def test_identical_teams_are_rejected(self):
        resp = self._post({'team_a': 'Sentinels', 'team_b': 'Sentinels'})
        self.assertEqual(resp.status_code, 400)

    def test_unknown_team_is_rejected(self):
        resp = self._post({
            'team_a': 'a team that does not exist anywhere',
            'team_b': 'another team that does not exist',
        })
        self.assertEqual(resp.status_code, 400)
        self.assertIn('error', resp.json())

    @requires_models
    def test_successful_prediction_shape(self):
        teams = predictor.get_all_teams()
        resp = self._post({'team_a': teams[0], 'team_b': teams[1]})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()

        for key in ('team_a_prob', 'team_b_prob', 'winner',
                    'confidence', 'all_agree', 'features'):
            self.assertIn(key, data)

        # Probabilities are complementary percentages.
        self.assertAlmostEqual(data['team_a_prob'] + data['team_b_prob'], 100.0, places=1)
        self.assertGreaterEqual(data['confidence'], 50.0)
        self.assertIn(data['winner'], (teams[0], teams[1]))

        # One breakdown row per model feature.
        self.assertEqual(len(data['features']), len(FEATURE_COLS))
        for f in data['features']:
            self.assertAlmostEqual(
                f['a_proportion'] + f['b_proportion'], 100.0, places=1
            )

    @requires_models
    def test_winner_matches_the_reported_probability(self):
        teams = predictor.get_all_teams()
        data = self._post({'team_a': teams[0], 'team_b': teams[1]}).json()
        expected = teams[0] if data['team_a_prob'] >= 50 else teams[1]
        self.assertEqual(data['winner'], expected)


@requires_models
class PredictorInternalTests(TestCase):

    def test_breakdown_covers_every_model_feature(self):
        teams = predictor.get_all_teams()
        result = predictor.predict_match(teams[0], teams[1])
        by_key = {f['key'] for f in result['features']}
        self.assertEqual(by_key, set(FEATURE_COLS))

    def test_probabilities_are_valid(self):
        teams = predictor.get_all_teams()
        result = predictor.predict_match(teams[0], teams[1])
        for key in ('lr_prob_a', 'rf_prob_a', 'xgb_prob_a', 'team_a_prob'):
            self.assertTrue(0.0 <= result[key] <= 100.0, f'{key}={result[key]}')

    def test_ensemble_is_the_mean_of_the_three_models(self):
        teams = predictor.get_all_teams()
        r = predictor.predict_match(teams[0], teams[1])
        mean = np.mean([r['lr_prob_a'], r['rf_prob_a'], r['xgb_prob_a']])
        self.assertAlmostEqual(r['team_a_prob'], mean, places=1)

    def test_all_agree_flag(self):
        teams = predictor.get_all_teams()
        r = predictor.predict_match(teams[0], teams[1])
        votes = {p >= 50 for p in (r['lr_prob_a'], r['rf_prob_a'], r['xgb_prob_a'])}
        self.assertEqual(r['all_agree'], len(votes) == 1)

    def test_analysis_metrics_are_plausible(self):
        m = predictor.get_analysis_metrics()
        self.assertEqual(m['total_count'],
                         m['train_count'] + m['val_count'] + m['test_count'])
        names = [r['name'] for r in m['metrics']]
        self.assertIn('Ensemble (average)', names)
        for row in m['metrics']:
            # A model with no signal would sit at 0.5; require better than that.
            self.assertGreater(float(row['auc']), 0.5)
            self.assertLessEqual(float(row['auc']), 1.0)

    def test_metric_highlighting_picks_exactly_one_winner_per_column(self):
        m = predictor.get_analysis_metrics()
        for key in ('accuracy', 'auc', 'f1', 'logloss', 'brier'):
            flagged = [r for r in m['metrics'] if r[f'{key}_best']]
            self.assertEqual(len(flagged), 1, f'{key}: {len(flagged)} flagged best')
