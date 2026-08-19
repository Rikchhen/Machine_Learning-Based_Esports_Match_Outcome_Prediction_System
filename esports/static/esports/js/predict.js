'use strict';

/* ── Helpers ────────────────────────────────────────────────── */
function csrfToken() {
  const m = document.cookie.match(/csrftoken=([^;]+)/);
  return m ? m[1] : '';
}

/* Percentages always render to 1 dp. JSON drops trailing zeros (13.0 -> 13),
   so formatting both sides of a split keeps them visually consistent. */
function pct(val) {
  return Number(val).toFixed(1);
}

function fmt(val, type) {
  if (val === null || val === undefined || (typeof val === 'number' && isNaN(val))) return '—';
  switch (type) {
    case 'float1': return Number(val).toFixed(1);
    case 'float2': return Number(val).toFixed(2);
    case 'pct':    return (Number(val) * 100).toFixed(1) + '%';
    case 'int':    return Math.round(val).toString();
    default:       return String(val);
  }
}

/* ── Build result HTML ──────────────────────────────────────── */
function buildResults(d) {
  const agreeBadge = d.all_agree
    ? `<span class="agree-badge agree">&#10003; All three models agree &mdash; ${esc(d.winner)} wins</span>`
    : `<span class="agree-badge disagree">&#9888; Models disagree &mdash; check breakdown below</span>`;

  // Feature rows (bars start at 0, animated in animateBars())
  const featureRows = d.features.map(f => {
    const aVal = fmt(f.a_val, f.fmt);
    const bVal = fmt(f.b_val, f.fmt);
    const aCls = 'feature-val' + (f.a_wins ? ' win-a' : '');
    const bCls = 'feature-val' + (f.a_wins ? '' : ' win-b');
    return `
      <div class="feature-row">
        <div class="feature-label">${esc(f.label)}</div>
        <div class="${aCls}">${aVal}</div>
        <div class="feat-bar">
          <div class="feat-bar-a" data-w="${f.a_proportion}" style="width:0%"></div>
          <div class="feat-bar-b" data-w="${f.b_proportion}" style="width:0%"></div>
        </div>
        <div class="${bCls}">${bVal}</div>
      </div>`;
  }).join('');

  const aName = esc(d.team_a);
  const bName = esc(d.team_b);
  const aTag  = aName.split(' ')[0].toUpperCase();
  const bTag  = bName.split(' ')[0].toUpperCase();
  const lrA   = pct(d.lr_prob_a),  lrB = pct(100 - d.lr_prob_a);
  const rfA   = pct(d.rf_prob_a),  rfB = pct(100 - d.rf_prob_a);
  const xgA   = pct(d.xgb_prob_a), xgB = pct(100 - d.xgb_prob_a);
  const ensA  = pct(d.team_a_prob), ensB = pct(d.team_b_prob);

  return `
    <!-- Match header -->
    <div class="match-header">
      <div class="match-teams">
        <span class="team-a-name">${aName}</span>
        <span class="vs">VS</span>
        <span class="team-b-name">${bName}</span>
      </div>
      <div class="match-meta">ML PREDICTION &nbsp;&middot;&nbsp; ENSEMBLE OF 3 MODELS</div>
    </div>

    <!-- Ensemble bar -->
    <div class="card ensemble-bar-card section-gap">
      <p class="card-title">Win Probability</p>
      <div class="ens-label">
        <span>${aName}</span>
        <span>${bName}</span>
      </div>
      <div class="ens-bar">
        <div class="ens-a" data-w="${d.team_a_prob}" style="width:0%">${ensA}%</div>
        <div class="ens-b" data-w="${d.team_b_prob}" style="width:0%">${ensB}%</div>
      </div>
      <div class="winner-label">
        Predicted winner: <strong>${esc(d.winner)}</strong> &nbsp;&middot;&nbsp; ${pct(d.confidence)}% confidence
      </div>
      <div style="text-align:center;margin-top:10px">${agreeBadge}</div>
    </div>

    <!-- Per-model breakdown -->
    <div class="card section-gap">
      <p class="card-title">Model Breakdown</p>

      <div class="model-row">
        <div class="model-name">Logistic Regression</div>
        <div class="model-bar">
          <div class="model-bar-a" data-w="${d.lr_prob_a}" style="width:0%"></div>
          <div class="model-bar-b" data-w="${lrB}" style="width:0%"></div>
        </div>
        <div class="model-pcts">
          <span class="pct-a">${lrA}%</span> &nbsp;·&nbsp; <span class="pct-b">${lrB}%</span>
        </div>
      </div>

      <div class="model-row">
        <div class="model-name">Random Forest</div>
        <div class="model-bar">
          <div class="model-bar-a" data-w="${d.rf_prob_a}" style="width:0%"></div>
          <div class="model-bar-b" data-w="${rfB}" style="width:0%"></div>
        </div>
        <div class="model-pcts">
          <span class="pct-a">${rfA}%</span> &nbsp;·&nbsp; <span class="pct-b">${rfB}%</span>
        </div>
      </div>

      <div class="model-row">
        <div class="model-name">XGBoost</div>
        <div class="model-bar">
          <div class="model-bar-a" data-w="${d.xgb_prob_a}" style="width:0%"></div>
          <div class="model-bar-b" data-w="${xgB}" style="width:0%"></div>
        </div>
        <div class="model-pcts">
          <span class="pct-a">${xgA}%</span> &nbsp;·&nbsp; <span class="pct-b">${xgB}%</span>
        </div>
      </div>
    </div>

    <!-- Feature comparison -->
    <div class="card">
      <p class="card-title">
        Stat Comparison
        <span style="font-size:11px;font-weight:400;color:var(--text-muted);margin-left:8px">
          Highlighted = advantage &nbsp;&middot;&nbsp; Pre-match career rolling averages
        </span>
      </p>
      <div>
        <div class="feature-row header-row">
          <div class="feature-label" style="font-size:11px;letter-spacing:1px;color:var(--text-muted)">STATISTIC</div>
          <div style="font-size:11px;letter-spacing:1px;color:var(--red);text-align:center">${aTag}</div>
          <div></div>
          <div style="font-size:11px;letter-spacing:1px;color:var(--blue);text-align:center">${bTag}</div>
        </div>
        ${featureRows}
      </div>
    </div>
  `;
}

function esc(str) {
  return String(str)
    .replace(/&/g,'&amp;').replace(/</g,'&lt;')
    .replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

function animateBars(container) {
  requestAnimationFrame(() => {
    setTimeout(() => {
      container.querySelectorAll('[data-w]').forEach(el => {
        el.style.width = el.dataset.w + '%';
      });
    }, 60);
  });
}

/* ── Main predict flow ──────────────────────────────────────── */
async function runPrediction(teamA, teamB) {
  const results = document.getElementById('results');
  results.classList.remove('hidden');
  results.innerHTML = `
    <div class="loading">
      <div class="spinner"></div>
      <p>Running prediction models&hellip;</p>
    </div>`;

  try {
    const resp = await fetch('/api/predict/', {
      method:  'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-CSRFToken':  csrfToken(),
      },
      body: JSON.stringify({ team_a: teamA, team_b: teamB }),
    });

    const data = await resp.json();

    if (!resp.ok) {
      results.innerHTML = `<div class="error-box">${esc(data.error || 'Prediction failed. Please try again.')}</div>`;
      return;
    }

    results.innerHTML = buildResults(data);
    animateBars(results);
    results.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

  } catch (err) {
    results.innerHTML = `<div class="error-box">Network error: ${esc(err.message)}</div>`;
  }
}

/* ── Bootstrap ──────────────────────────────────────────────── */
function showFormError(message) {
  const results = document.getElementById('results');
  results.classList.remove('hidden');
  results.innerHTML = `<div class="error-box">${esc(message)}</div>`;
}

document.addEventListener('DOMContentLoaded', () => {
  const form = document.getElementById('predict-form');
  const btn  = document.getElementById('predict-btn');
  if (!form) return;

  const selA = document.getElementById('team-a');
  const selB = document.getElementById('team-b');

  // Prefill from ?team_a=…&team_b=… — used by the /teams/ page links, and
  // makes a chosen matchup shareable as a URL. Assigning an unknown value
  // leaves the select blank, which is what we want for a stale link.
  const params = new URLSearchParams(location.search);
  [['team_a', selA], ['team_b', selB]].forEach(([param, sel]) => {
    const val = params.get(param);
    if (val && sel) sel.value = val;
  });

  form.addEventListener('submit', async e => {
    e.preventDefault();
    const teamA = selA.value.trim();
    const teamB = selB.value.trim();

    if (!teamA || !teamB) {
      showFormError('Please select both teams before predicting.');
      (teamA ? selB : selA).focus();
      return;
    }
    if (teamA === teamB) {
      showFormError('Please select two different teams.');
      selB.focus();
      return;
    }

    btn.disabled    = true;
    btn.textContent = 'Predicting…';
    try {
      await runPrediction(teamA, teamB);
    } finally {
      btn.disabled    = false;
      btn.textContent = 'Predict Match';
    }
  });
});
