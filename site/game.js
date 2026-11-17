/* Matchup page: reweights every unit live. Mirrors engine/model.py exactly. */
(async function () {
  const id = new URLSearchParams(location.search).get('id');
  const [G, W] = await Promise.all([
    fetch(`data/games/${encodeURIComponent(id)}.json`, { cache: 'no-cache' }).then(r => r.json()),
    fetch('data/week.json', { cache: 'no-cache' }).then(r => r.json()),
  ]);
  const META = W.meta;
  const METRICS = META.metrics, LABELS = META.labels, COUNT = META.count_col, HIB = META.higher_is_better;
  const game = G.game, HOME = game.home, AWAY = game.away;
  const TUNED = G.params, PM = G.points_model, BEST = G.best_a || { a_opp: 0, a_rec: 0, a_set: 0, a_ros: 0 };
  const $ = s => document.getElementById(s);
  const esc = s => String(s).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

  const state = {
    a: { a_opp: TUNED.a_opp, a_rec: TUNED.a_rec, a_set: TUNED.a_set, a_ros: TUNED.a_ros },
    view: 'overlay',
    unit: `${HOME}|defense`,
    qb: { [HOME]: G.units[`${HOME}|offense`].expected_qb, [AWAY]: G.units[`${AWAY}|offense`].expected_qb },
    plays: {},   // team -> { rb: bool expected to play, wr: bool }
    method: false,
  };

  for (const t of [HOME, AWAY]) {
    const k = G.units[`${t}|offense`].key || {};
    state.plays[t] = { rb: k.rb_expected !== false, wr: k.wr_expected !== false };
  }

  // ---------- math (mirror of engine/model.py)
  const clamp = (x, lo, hi) => Math.min(hi, Math.max(lo, x));
  function factors(g, team, side) {
    const sim = clamp(Math.exp(-(g.dist ** 2) / (2 * TUNED.sim_bw ** 2)), 1e-4, 1);
    const rec = clamp(Math.pow(0.5, g.weeks_ago / TUNED.half_life), 1e-4, 1);
    const wx = g.wx_d2 == null ? 1 : Math.exp(-g.wx_d2 / 2);
    const set = clamp(g.site_base * wx, 1e-4, 1);
    let ros = 1;
    if (side === 'offense') {
      const key = G.units[`${team}|offense`].key || {};
      const qbm = state.qb[team] && g.qb === state.qb[team];
      const rbm = !key.rb || g.rb_played === state.plays[team].rb;
      const wrm = !key.wr || g.wr_played === state.plays[team].wr;
      ros = (qbm ? 1 : TUNED.qb_miss) * (rbm ? 1 : TUNED.skill_miss) * (wrm ? 1 : TUNED.skill_miss);
    }
    return { sim, rec, set, ros };
  }
  function unitProfile(key, a) {
    const [team, side] = key.split('|');
    const U = G.units[key];
    const games = U.games;
    const f = games.map(g => factors(g, team, side));
    const w = f.map(x => Math.pow(x.sim, a.a_opp) * Math.pow(x.rec, a.a_rec) * Math.pow(x.set, a.a_set) * Math.pow(x.ros, a.a_ros));
    const weighted = {}, season = {}, final = {};
    for (const m of METRICS) {
      let sw = 0, swx = 0, sn = 0, snx = 0;
      games.forEach((g, i) => {
        const x = g.m[m], n = g[COUNT[m]];
        if (x == null || !(n > 0)) return;
        sw += w[i] * n; swx += w[i] * n * x; sn += n; snx += n * x;
      });
      weighted[m] = sw > 0 ? swx / sw : null;
      season[m] = sn > 0 ? snx / sn : null;
    }
    const sumw = w.reduce((s, v) => s + v, 0), sumw2 = w.reduce((s, v) => s + v * v, 0);
    const n_eff = sumw2 > 0 ? sumw * sumw / sumw2 : 0;
    const lam = n_eff / (n_eff + TUNED.k);
    for (const m of METRICS) {
      const b = U.baseline[m], x = weighted[m];
      final[m] = (x != null && b != null) ? lam * x + (1 - lam) * b : (x != null ? x : b);
    }
    if (!games.length) { Object.assign(weighted, U.baseline); Object.assign(season, U.baseline); }
    // Opponent-adjusted EPA/play for the projection (mirror of model.unit_profile epa_proj)
    let an = 0, ad = 0;
    games.forEach((g, i) => { const x = g.m.epa_play - TUNED.adj * (g.opp_level - G.lg_epa); an += w[i] * g.plays * x; ad += w[i] * g.plays; });
    const adjW = ad > 0 ? an / ad : null, bE = U.baseline.epa_play;
    const epa_proj = adjW == null ? bE : (bE == null ? adjW : lam * adjW + (1 - lam) * bE);
    const tot = w.reduce((s, v, i) => s + v * games[i].plays, 0);
    const share = w.map((v, i) => tot > 0 ? v * games[i].plays / tot : 0);
    return { weighted, season, final, n_eff, lam, w, share, f, epa_proj };
  }
  function pct(v, side, m) {
    const arr = G.refs[`${side}|${m}`];
    if (v == null || !arr || !arr.length) return null;
    let lo = 0, hi = arr.length;
    while (lo < hi) { const mid = (lo + hi) >> 1; if (arr[mid] < v) lo = mid + 1; else hi = mid; }
    const frac = lo / arr.length;
    return 100 * (HIB[`${side}|${m}`] ? frac : 1 - frac);
  }
  const erf = x => { const s = Math.sign(x); x = Math.abs(x); const t = 1 / (1 + 0.3275911 * x);
    const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x);
    return s * y; };
  function projection(a) {
    const P = {};
    for (const k of Object.keys(G.units)) P[k] = unitProfile(k, a);
    const loc = game.neutral ? 0 : 1;
    const pts = (team, opp, l) => {
      const e = P[`${team}|offense`].epa_proj + P[`${opp}|defense`].epa_proj - G.lg_epa;
      return PM.a + PM.b * e + PM.c * (G.plays[team] - 65) + PM.hfa * l;
    };
    const h = pts(HOME, AWAY, loc), aw = pts(AWAY, HOME, -loc);
    const n = Math.min(...Object.values(P).map(p => p.n_eff));
    const sd = PM.sd0 * Math.sqrt(1 + PM.sd_k / Math.max(n, 0.5));
    const wp = 0.5 * (1 + erf((h - aw) / sd / Math.SQRT2));
    return { P, home: h, away: aw, wp, sd, n };
  }

  // ---------- static header
  const kick = new Date(game.start);
  $('eyebrow').textContent = `${game.season} · Week ${game.week} · ${kick.toLocaleString([], { weekday: 'short', month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit' })}`;
  $('title').innerHTML = `<span class="away">${esc(AWAY)}</span> ${game.neutral ? 'vs' : 'at'} <span class="home">${esc(HOME)}</span>`;
  document.title = `${AWAY} ${game.neutral ? 'vs' : 'at'} ${HOME} · Weighted Matchups`;
  const chips = [];
  chips.push(game.neutral ? 'Neutral site' : `${HOME} at home`);
  if (game.venue) chips.push(game.venue);
  if (game.weather) chips.push(game.weather.indoors ? 'Indoors' : `Forecast: ${Math.round(game.weather.temp)}°F, wind ${Math.round(game.weather.wind || 0)} mph${game.weather.precip > 0 ? ', precip' : ''}`);
  $('chips').innerHTML = chips.map(c => `<span class="chip">${esc(c)}</span>`).join('') +
    (G.common_opponents.length ? `<span class="chip info">Common opponents: ${esc(G.common_opponents.join(', '))}</span>` : '');
  $('pAwayName').textContent = AWAY; $('pHomeName').textContent = HOME;
  const sp = game.spread_home;
  $('pMarket').textContent = sp == null ? 'n/a' : (Math.abs(sp) < 0.25 ? 'Pick ’em' : (sp < 0 ? `${HOME} ${sp}` : `${AWAY} −${sp}`));

  function formChips(team, el, lbl) {
    $(lbl).textContent = `${team}: last ${G.form[team].length} games`;
    $(el).innerHTML = G.form[team].map(f => `<span class="${f.result}">${f.date} ${f.result} ${f.pts}–${f.opp_pts} ${f.site === 'away' ? '@' : (f.site === 'neutral' ? 'vs' : 'vs')} ${esc(f.opp)}</span>`).join('');
  }
  formChips(AWAY, 'formAway', 'formAwayLbl'); formChips(HOME, 'formHome', 'formHomeLbl');

  // ---------- methodology
  function methodHTML() {
    const bt = META.backtest;
    let table = '<p>Backtest results will appear after the first tuning run.</p>';
    if (bt) {
      const r = (x, d = 2) => x == null ? '–' : x.toFixed(d);
      const tw = bt.weighted.test, te = bt.equal_weight.test, mk = bt.market;
      table = `<table><thead><tr><th>Held-out ${bt.test_seasons.join(', ')} games</th><th>Games</th><th>Avg miss (pts)</th><th>Winner right</th><th>Brier</th></tr></thead><tbody>
        <tr><td>Backtest-best weights</td><td>${tw.games}</td><td>${r(tw.mae, 1)}</td><td>${r(tw.winner_acc * 100, 0)}%</td><td>${r(tw.brier, 3)}</td></tr>
        ${bt.site_default ? `<tr><td>Site default sliders (all Normal)</td><td>${bt.site_default.test.games}</td><td>${r(bt.site_default.test.mae, 1)}</td><td>${r(bt.site_default.test.winner_acc * 100, 0)}%</td><td>${r(bt.site_default.test.brier, 3)}</td></tr>` : ''}
        <tr><td>Equal-weight model</td><td>${te.games}</td><td>${r(te.mae, 1)}</td><td>${r(te.winner_acc * 100, 0)}%</td><td>${r(te.brier, 3)}</td></tr>
        <tr><td>Market spread (same games)</td><td>${mk.test.games}</td><td>${r(mk.test.mae, 1)}</td><td>–</td><td>–</td></tr>
        <tr><td>Backtest-best (same games)</td><td>${mk.ours_same_games.games}</td><td>${r(mk.ours_same_games.mae, 1)}</td><td>${r(mk.ours_same_games.winner_acc * 100, 0)}%</td><td>${r(mk.ours_same_games.brier, 3)}</td></tr>
      </tbody></table>
      <div style="font-size:13px;color:#e2e8f0"><b style="display:inline">What the backtest says:</b> once each game's EPA is opponent-adjusted, extra weight on similar opponents, recency or setting has not measurably improved margin predictions on past games; the key-player (roster) factor helped slightly. Backtest-best strengths: similar opponents ${BEST.a_opp}, recency ${BEST.a_rec}, setting ${BEST.a_set}, QB ${BEST.a_ros}. The sliders start at Normal so you can see the weighted view; use “Backtest-best” for the most accurate projection so far.</div>
      <div style="font-size:12px;color:#cbd5e1">Tuned on ${bt.train_seasons.join(' and ')} only; ${bt.test_seasons.join(', ')} was never used for tuning. Avg miss = mean absolute error of the predicted margin. Brier = mean squared error of win probability (lower is better; 0.25 = coin flip).</div>`;
    }
    return `<h3>How the projection works</h3>
      <div class="grid">
        <div><b>1. Weight every past game</b>Each of a unit's games this season gets a weight from four factors multiplied together: how similar the opponent unit was to this week's opponent unit (opponent-adjusted EPA, success rate, pass rate, explosiveness, rush and pass EPA), how recent it was (half-life ${TUNED.half_life} weeks), whether the setting matches (home/away/neutral and forecast weather), and whether the expected QB, top RB and top WR played (a different QB counts ×${TUNED.qb_miss}, each RB/WR mismatch ×${TUNED.skill_miss}, before the Roster slider). RB/WR status comes from box scores: a player with no carries or catches is treated as not playing.</div>
        <div><b>2. Unit-on-unit</b>A defense is judged mostly on games against offenses like the one it faces this week, and an offense on games against similar defenses.</div>
        <div><b>3. Weighted profiles</b>Per-play stats (EPA, success rate, explosiveness, havoc…) are averaged across games using weight × plays. Garbage time is removed.</div>
        <div><b>4. Confidence blend</b>Effective games = (Σw)² ÷ Σw². The profile = λ × weighted + (1 − λ) × prior-season average, with λ = n ÷ (n + ${TUNED.k}). Early in the season, or when weight piles onto a few games, the prior dominates.</div>
        <div><b>5. Points</b>For the projection, each game's EPA/play is first opponent-adjusted (minus how far that opponent's unit sits from the FBS average, at strength ${TUNED.adj}). Expected EPA/play = offense EPA + opposing defense EPA allowed − FBS average. Points = ${PM.a.toFixed(1)} + ${PM.b.toFixed(1)} × expected EPA + ${PM.c.toFixed(2)} × (projected plays − 65) ± ${PM.hfa.toFixed(2)} home field, fit by regression on past seasons.</div>
        <div><b>6. Win probability</b>The margin is treated as a normal distribution around the projection with SD ${PM.sd0.toFixed(1)} × √(1 + ${PM.sd_k.toFixed(2)} ÷ effective games), fit on past results.</div>
      </div>
      ${table}
      <div style="font-size:12px;color:#cbd5e1">Factor strengths, half-life, similarity bandwidth and the confidence constant were chosen by backtest. Sliders change the strengths live. Data: CollegeFootballData.com play-by-play (EPA = CFBD PPA), schedules, weather and consensus lines.</div>`;
  }
  $('method').innerHTML = methodHTML();
  $('methodBtn').addEventListener('click', () => {
    state.method = !state.method;
    $('method').hidden = !state.method;
    $('methodBtn').setAttribute('aria-expanded', String(state.method));
    $('methodBtn').textContent = state.method ? 'Hide methodology' : 'How this projection works';
  });

  // ---------- controls
  const SL = [
    ['a_opp', 'Similar opponents', 'Favor games vs opponent units that look like this week’s.'],
    ['a_rec', 'Recency', 'Recent weeks count more.'],
    ['a_set', 'Setting', 'Home/away match and forecast weather.'],
    ['a_ros', 'Roster (QB)', 'Games started by the expected QB.'],
  ];
  const lvl = v => v === 0 ? 'Off' : v < 1 ? 'Light' : v === 1 ? 'Normal' : v < 2 ? 'Strong' : 'Very strong';
  $('sliders').innerHTML = SL.map(([k, l, h]) => `
    <div class="slider"><div class="top"><label for="s_${k}">${l}</label><span class="pill" id="v_${k}"></span></div>
    <input id="s_${k}" type="range" min="0" max="3" step="0.25"><div class="small">${h}</div></div>`).join('');
  SL.forEach(([k]) => $(`s_${k}`).addEventListener('input', e => { state.a[k] = parseFloat(e.target.value); render(); }));
  $('reset').addEventListener('click', () => { Object.assign(state.a, { a_opp: TUNED.a_opp, a_rec: TUNED.a_rec, a_set: TUNED.a_set, a_ros: TUNED.a_ros }); render(); });
  $('best').addEventListener('click', () => { Object.assign(state.a, BEST); render(); });
  $('equal').addEventListener('click', () => { Object.assign(state.a, { a_opp: 0, a_rec: 0, a_set: 0, a_ros: 0 }); render(); });

  $('qbs').innerHTML = [AWAY, HOME].map(t => {
    const U = G.units[`${t}|offense`], k = U.key || {};
    const opts = U.qbs.map(q => `<option value="${esc(q)}">${esc(q)}${q === U.expected_qb ? ' (most recent starter)' : ''}</option>`).join('');
    const status = (pos, name, detail) => !name ? '' : `<div class="row2"><span><b>${pos}</b> ${esc(name)} <span class="small">${detail}</span></span>
      <select data-team="${esc(t)}" data-pos="${pos.toLowerCase()}" aria-label="${esc(t)} ${pos} ${esc(name)} status">
        <option value="1">Expected to play</option><option value="0">Out</option></select></div>`;
    return `<div class="kp"><div style="font-weight:700;color:${t === HOME ? 'var(--home)' : 'var(--away)'}">${esc(t)}</div>
      <div class="row2"><label for="qb_${esc(t)}"><b>QB</b></label>
      <select id="qb_${esc(t)}" data-team="${esc(t)}" data-pos="qb">${opts}<option value="__other">Someone else</option></select></div>
      ${status('RB', k.rb, k.rb_carries ? `(${k.rb_carries} carries)` : '')}
      ${status('WR', k.wr, k.wr_yards ? `(${k.wr_yards} rec yds)` : '')}</div>`;
  }).join('');
  document.querySelectorAll('#qbs select').forEach(s => {
    const t = s.dataset.team, pos = s.dataset.pos;
    if (pos === 'qb') {
      s.value = state.qb[t] || '__other';
      s.addEventListener('change', () => { state.qb[t] = s.value === '__other' ? '__none__' : s.value; render(); });
    } else {
      s.value = state.plays[t][pos] ? '1' : '0';
      s.addEventListener('change', () => { state.plays[t][pos] = s.value === '1'; render(); });
    }
  });

  document.querySelectorAll('#viewseg button').forEach(b => b.addEventListener('click', () => { state.view = b.dataset.v; render(); }));
  const UNITS = [`${AWAY}|offense`, `${AWAY}|defense`, `${HOME}|offense`, `${HOME}|defense`];
  $('unitseg').innerHTML = UNITS.map(u => { const [t, s] = u.split('|'); return `<button data-u="${esc(u)}">${esc(t)} ${s === 'offense' ? 'O' : 'D'}</button>`; }).join('');
  document.querySelectorAll('#unitseg button').forEach(b => b.addEventListener('click', () => { state.unit = b.dataset.u; render(); }));

  // ---------- radar
  const N = METRICS.length, CX = 300, CY = 275, R = 175;
  const pt = (v, i) => { const a = (-90 + 360 / N * i) * Math.PI / 180; return [CX + R * v / 100 * Math.cos(a), CY + R * v / 100 * Math.sin(a)]; };
  const poly = vals => vals.map((v, i) => pt(v == null ? 0 : v, i).map(x => x.toFixed(1)).join(',')).join(' ');
  function radarSVG(title, offKey, defKey, P) {
    const [ot] = offKey.split('|'), [dt] = defKey.split('|');
    const cOff = ot === HOME ? 'var(--home)' : 'var(--away)', fOff = ot === HOME ? 'var(--home-fill)' : 'var(--away-fill)';
    const cDef = dt === HOME ? 'var(--home)' : 'var(--away)', fDef = dt === HOME ? 'var(--home-fill)' : 'var(--away-fill)';
    const vals = (key, side, which) => METRICS.map(m => pct(P[key][which][m], side, m));
    let s = `<svg viewBox="0 0 600 560" role="img" aria-label="${esc(title)} percentile radar">`;
    [100, 75, 50, 25].forEach((r, i) => s += `<polygon points="${poly(Array(N).fill(r))}" style="fill:${i === 0 ? '#fafafa' : 'none'};stroke:${r === 50 ? '#d1d5db' : '#e5e7eb'};stroke-width:1"/>`);
    for (let i = 0; i < N; i++) { const [x, y] = pt(100, i); s += `<line x1="${CX}" y1="${CY}" x2="${x.toFixed(1)}" y2="${y.toFixed(1)}" style="stroke:#e5e7eb;stroke-width:1"/>`; }
    [50, 75, 100].forEach(r => { const [x, y] = pt(r, 0); s += `<text x="${x + 4}" y="${y + 12}" style="font:10px Inter,sans-serif;fill:#6b7280">${r}</text>`; });
    if (state.view !== 'weighted') {
      s += `<polygon points="${poly(vals(defKey, 'defense', 'season'))}" style="fill:none;stroke:${cDef};stroke-width:2.5;stroke-dasharray:6 5"/>`;
      s += `<polygon points="${poly(vals(offKey, 'offense', 'season'))}" style="fill:none;stroke:${cOff};stroke-width:2.5;stroke-dasharray:6 5"/>`;
    }
    if (state.view !== 'season') {
      s += `<polygon points="${poly(vals(defKey, 'defense', 'final'))}" style="fill:${fDef};stroke:${cDef};stroke-width:2.5"/>`;
      s += `<polygon points="${poly(vals(offKey, 'offense', 'final'))}" style="fill:${fOff};stroke:${cOff};stroke-width:2.5"/>`;
    }
    METRICS.forEach((m, i) => {
      const [x, y] = pt(118, i); const anchor = Math.abs(x - CX) < 8 ? 'middle' : (x > CX ? 'start' : 'end');
      s += `<text x="${x.toFixed(1)}" y="${(y + 4).toFixed(1)}" text-anchor="${anchor}" style="font:600 12px Inter,sans-serif;fill:#374151">${esc(LABELS[m])}</text>`;
    });
    return s + '</svg>';
  }

  // ---------- per-unit games table
  const r0 = x => Math.round(x);
  function unitTable(key, P) {
    const [t, side] = key.split('|');
    const U = G.units[key], k = U.key || {};
    const color = t === HOME ? 'var(--home)' : 'var(--away)';
    const maxS = Math.max(...P.share, 1e-9);
    const rows = U.games.map((g, i) => ({ g, i })).sort((a, b) => b.g.week - a.g.week);
    const body = rows.map(({ g, i }) => {
      const f = P.f[i], tags = [];
      if (g.common) tags.push(['Common opponent', 'good']);
      if (f.sim >= 0.7) tags.push([side === 'defense' ? 'Similar offense' : 'Similar defense', 'good']);
      else if (f.sim < 0.3) tags.push([side === 'defense' ? 'Different offense' : 'Different defense', 'warn']);
      if (f.rec >= 0.75) tags.push(['Recent', 'good']); else if (f.rec < 0.35) tags.push(['Older', 'warn']);
      if (g.site_base >= 1) tags.push([`Same site (${g.site})`, 'good']); else tags.push([`${g.site[0].toUpperCase()}${g.site.slice(1)}`, '']);
      if (g.wx_d2 != null && g.wx_d2 > 1) tags.push(['Different weather', 'warn']);
      if (side === 'offense') {
        if (!(state.qb[t] && g.qb === state.qb[t])) tags.push(['Different QB', 'warn']);
        if (k.rb && g.rb_played !== state.plays[t].rb) tags.push([g.rb_played ? `${k.rb} played` : `No ${k.rb}`, 'warn']);
        if (k.wr && g.wr_played !== state.plays[t].wr) tags.push([g.wr_played ? `${k.wr} played` : `No ${k.wr}`, 'warn']);
      }
      if (g.fcs) tags.push(['FCS opponent', 'warn']);
      const res = g.pts > g.opp_pts ? 'W' : (g.pts < g.opp_pts ? 'L' : 'T');
      return `<tr><td>${g.week}</td><td style="font-weight:600">${esc(g.opponent)}</td>
        <td class="${res === 'W' ? 'res-w' : 'res-l'}">${res} ${g.pts}–${g.opp_pts}</td><td>${esc(g.site)}</td><td>${esc((side === 'offense' ? g.qb : g.opp_qb) || '')}</td>
        <td><div class="wbar"><div class="track"><div class="fill" style="width:${(P.share[i] / maxS * 100).toFixed(1)}%;background:${color}"></div></div><span class="pct">${r0(P.share[i] * 100)}%</span></div></td>
        <td>${tags.map(([x, c]) => `<span class="tag ${c}">${esc(x)}</span>`).join('')}</td></tr>`;
    }).join('');
    const facing = side === 'offense' ? `vs ${t === HOME ? AWAY : HOME} defense` : `vs ${t === HOME ? AWAY : HOME} offense`;
    return `<div class="card unitcard" style="display:flex;flex-direction:column;gap:10px;border-top:4px solid ${color}">
      <div class="row" style="justify-content:space-between;align-items:baseline;gap:8px">
        <h3>Games driving ${esc(t)}'s ${side === 'offense' ? 'offensive' : 'defensive'} profile <span class="small" style="font-weight:500">(${esc(facing)})</span></h3>
        <span class="small">${P.n_eff.toFixed(1)} effective games · ${r0(P.lam * 100)}% weighted</span></div>
      <div class="scroll"><table class="games"><thead><tr><th>Wk</th><th>Opponent</th><th>Result</th><th>Site</th><th>${side === 'offense' ? 'QB' : 'Opp QB'}</th><th>Weight share</th><th>Why</th></tr></thead><tbody>${body}</tbody></table></div></div>`;
  }

  // ---------- render
  function render() {
    SL.forEach(([k]) => { $(`s_${k}`).value = state.a[k]; $(`v_${k}`).textContent = lvl(state.a[k]) + (state.a[k] === BEST[k] ? ' · backtest-best' : ''); });
    document.querySelectorAll('#viewseg button').forEach(b => b.setAttribute('aria-pressed', String(b.dataset.v === state.view)));
    document.querySelectorAll('#unitseg button').forEach(b => b.setAttribute('aria-pressed', String(b.dataset.u === state.unit)));

    const pr = projection(state.a);
    const eq = projection({ a_opp: 0, a_rec: 0, a_set: 0, a_ros: 0 });
    $('pAway').textContent = r0(pr.away); $('pHome').textContent = r0(pr.home);
    const mg = pr.home - pr.away, rm = r0(mg);
    const lo = mg - 1.2816 * pr.sd, hi = mg + 1.2816 * pr.sd;
    const side = v => { const x = r0(v); return x === 0 ? 'even' : (x > 0 ? `${HOME} +${x}` : `${AWAY} +${-x}`); };
    $('pMargin').textContent = `${rm === 0 ? 'Pick ’em' : (rm > 0 ? `${HOME} by ${rm}` : `${AWAY} by ${-rm}`)} · 80% range: ${side(lo)} to ${side(hi)}`;
    const wa = r0((1 - pr.wp) * 100);
    $('wpAway').textContent = `${AWAY} ${wa}%`; $('wpHome').textContent = `${HOME} ${100 - wa}%`;
    $('wpA').style.width = `${wa}%`;
    $('wpbar').setAttribute('aria-label', `Win probability: ${AWAY} ${wa} percent, ${HOME} ${100 - wa} percent`);
    $('pEqual').textContent = `${AWAY} ${r0(eq.away)} – ${HOME} ${r0(eq.home)}`;
    $('pPlays').textContent = `${AWAY} ${r0(G.plays[AWAY])} · ${HOME} ${r0(G.plays[HOME])}`;

    const sw = (team, solid) => {
      const c = team === HOME ? 'var(--home)' : 'var(--away)', f = team === HOME ? 'var(--home-fill)' : 'var(--away-fill)';
      return `<svg width="34" height="18" aria-hidden="true"><polygon points="3,15 10,3 24,3 31,15" style="fill:${solid ? f : 'none'};stroke:${c};stroke-width:${solid ? 2.5 : 2};${solid ? '' : 'stroke-dasharray:5 4'}"/></svg>`;
    };
    $('legend').innerHTML = `<div class="rlegend" role="list">
      <div class="k ${state.view === 'season' ? 'off' : ''}">Weighted<small>Solid, filled · for this matchup</small></div>
      <div class="items ${state.view === 'season' ? 'off' : ''}"><span>${sw(AWAY, true)}${esc(AWAY)}</span><span>${sw(HOME, true)}${esc(HOME)}</span></div>
      <div class="k ${state.view === 'weighted' ? 'off' : ''}">Unweighted<small>Dashed outline · plain season average</small></div>
      <div class="items ${state.view === 'weighted' ? 'off' : ''}"><span>${sw(AWAY, false)}${esc(AWAY)}</span><span>${sw(HOME, false)}${esc(HOME)}</span></div></div>`;
    $('radars').innerHTML = `
      <div class="radar"><h2 style="font-size:18px">${esc(AWAY)} Offense vs ${esc(HOME)} Defense</h2>${radarSVG(`${AWAY} offense vs ${HOME} defense`, `${AWAY}|offense`, `${HOME}|defense`, pr.P)}</div>
      <div class="radar"><h2 style="font-size:18px">${esc(AWAY)} Defense vs ${esc(HOME)} Offense</h2>${radarSVG(`${AWAY} defense vs ${HOME} offense`, `${HOME}|offense`, `${AWAY}|defense`, pr.P)}</div>`;

    $('confRows').innerHTML = UNITS.map(u => { const p = pr.P[u]; const [t, s] = u.split('|');
      return `<div><div style="display:flex;justify-content:space-between;font-size:13px"><span><b>${esc(t)}</b> ${s}</span><span>${p.n_eff.toFixed(1)} eff. games · ${r0(p.lam * 100)}% weighted</span></div>
      <div class="bar"><div style="width:${r0(p.lam * 100)}%;background:${t === HOME ? 'var(--home)' : 'var(--away)'}"></div></div></div>`; }).join('');

    // games tables: one per unit, ordered like the radars
    const [ut, us] = state.unit.split('|');
    const P = pr.P[state.unit];
    const ORDER = [`${AWAY}|offense`, `${HOME}|defense`, `${HOME}|offense`, `${AWAY}|defense`];
    $('unitTables').innerHTML = ORDER.map(key => unitTable(key, pr.P[key])).join('');
    $('shiftTitle').textContent = `What the weighting changes: ${ut} ${us}`;
    $('shifts').innerHTML = METRICS.map(m => {
      const s = pct(P.season[m], us, m), f = pct(P.final[m], us, m);
      if (s == null || f == null) return '';
      const d = r0(f) - r0(s);
      return `<div class="shift"><span class="m">${esc(LABELS[m])}</span><div class="v"><span class="small">Season ${r0(s)}</span><span class="small">→</span><span class="f">${r0(f)}</span>
        <span class="delta ${d < 0 ? 'down' : 'up'}">${d > 0 ? '+' : d < 0 ? '−' : '±'}${Math.abs(d)}</span></div></div>`;
    }).join('');
  }
  render();
})();
