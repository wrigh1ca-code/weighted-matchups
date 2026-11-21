/* Build-your-own matchups: assembles the same data shape as data/games/<id>.json for any two FBS
   teams, from data/league.json and data/teams/<slug>.json.

   Only two weighting ingredients depend on the matchup, and both are recomputed here exactly as
   engine/model.unit_ingredients does:
     dist       distance between the opponent unit faced in each past game and this week's opponent
                unit (weighted RMS over the scaled similarity features; 3.0 when either is missing)
     site_base  SITE_MATCH[(this game's site, past game's site)], default 0.6
   The user can pick a weather scenario; wx_d2 then mirrors model._wx_dist2 against each past game's
   conditions. With no scenario, wx_d2 = null (no weather factor). */
(function () {
  const cache = {};
  const getJSON = url => cache[url] || (cache[url] = fetch(url, { cache: 'no-cache' }).then(r => {
    if (!r.ok) throw new Error(`${url}: ${r.status}`);
    return r.json();
  }));

  const loadLeague = () => getJSON('data/league.json');

  async function loadTeam(L, name) {
    const t = L.teams.find(x => x.team === name);
    if (!t) throw new Error(`Unknown team: ${name}`);
    return getJSON(`data/teams/${t.slug}.json`);
  }

  function dist(L, a, b) {
    if (!a || !b) return 3.0;
    const w = L.sim_weights;
    let s = 0, sw = 0;
    for (let i = 0; i < w.length; i++) {
      if (a[i] == null || b[i] == null) return 3.0;
      s += w[i] * (a[i] - b[i]) ** 2; sw += w[i];
    }
    return Math.sqrt(s / sw);
  }

  // Weather scenarios for hypothetical games (temp °F, wind mph, precip > 0 = wet)
  const WEATHER = {
    warm: { label: 'Sunny and warm', indoors: false, temp: 75, wind: 6, precip: 0 },
    cold: { label: 'Cold, dry', indoors: false, temp: 35, wind: 6, precip: 0 },
    warmwet: { label: 'Warm with rain', indoors: false, temp: 70, wind: 6, precip: 0.1 },
    coldwet: { label: 'Cold with rain or snow', indoors: false, temp: 35, wind: 6, precip: 0.1 },
    dome: { label: 'Indoors (dome)', indoors: true, temp: null, wind: null, precip: null },
  };
  function wxNorm(x) {
    if (!x) return null;
    if (x.indoors) return [70, 0, 0];
    if (x.temp == null) return null;
    return [x.temp, x.wind || 0, x.precip || 0];
  }
  function wxDist2(t, g) {   // mirror of model._wx_dist2
    const a = wxNorm(t), b = wxNorm(g);
    if (!a || !b) return null;
    return ((a[0] - b[0]) / 25) ** 2 + ((a[1] - b[1]) / 12) ** 2 + 0.5 * ((a[2] > 0) !== (b[2] > 0) ? 1 : 0);
  }

  const meanPlays = games => games.length ? games.reduce((s, g) => s + g.plays, 0) / games.length : 65;

  async function buildCustom(home, away, neutral, weatherKey) {
    const wx = WEATHER[weatherKey] || null;
    if (!home || !away || home === away) throw new Error('Pick two different teams.');
    const L = await loadLeague();
    const [TH, TA] = await Promise.all([loadTeam(L, home), loadTeam(L, away)]);
    const T = { [home]: TH, [away]: TA };
    const opp = { [home]: away, [away]: home };
    const site = t => neutral ? 'neutral' : (t === home ? 'home' : 'away');

    const opps = t => new Set(T[t].units.offense.games.map(g => g.opponent));
    const oh = opps(home), oa = opps(away);
    const common = [...oh].filter(x => oa.has(x) && x !== home && x !== away).sort();
    const commonSet = new Set(common);

    const units = {};
    for (const t of [home, away]) {
      for (const side of ['offense', 'defense']) {
        const other = side === 'offense' ? 'defense' : 'offense';
        const target = L.vec[`${opp[t]}|${other}`];
        const U = T[t].units[side];
        units[`${t}|${side}`] = {
          ...U,
          games: U.games.map(g => ({
            ...g,
            dist: dist(L, L.vec[`${g.opponent}|${other}`], target),
            site_base: L.site_match[`${site(t)}|${g.site}`] ?? 0.6,
            wx_d2: wxDist2(wx, g.wx),
            common: commonSet.has(g.opponent),
          })),
        };
      }
    }
    const plays = {};
    for (const t of [home, away]) {
      plays[t] = (meanPlays(units[`${t}|offense`].games) + meanPlays(units[`${opp[t]}|defense`].games)) / 2;
    }
    return {
      game: {
        id: null, custom: true, season: L.season, week: null, start: L.asof, home, away, neutral: !!neutral,
        venue: null, home_conf: TH.conference, away_conf: TA.conference,
        weather: wx ? { indoors: wx.indoors, temp: wx.temp, wind: wx.wind, precip: wx.precip } : null, weather_key: wx ? weatherKey : null, spread_home: null, total: null,
      },
      common_opponents: common,
      form: { [home]: TH.form, [away]: TA.form },
      units, refs: L.refs, lg_epa: L.lg_epa, plays,
      params: L.params, points_model: L.points_model, best_a: L.best_a,
    };
  }

  // Recently built matchups (this browser only; storage can be unavailable, so never let it throw)
  const KEY = 'wm.recent.v1';
  function recent() {
    try { return JSON.parse(localStorage.getItem(KEY) || '[]'); } catch (e) { return []; }
  }
  function remember(m) {
    try {
      const same = x => x.home === m.home && x.away === m.away && !!x.neutral === !!m.neutral && (x.wx || null) === (m.wx || null);
      const list = recent().filter(x => !same(x));
      list.unshift({ home: m.home, away: m.away, neutral: !!m.neutral, wx: m.wx || null });
      localStorage.setItem(KEY, JSON.stringify(list.slice(0, 8)));
    } catch (e) { /* ignore */ }
  }
  const url = m => `game.html?away=${encodeURIComponent(m.away)}&home=${encodeURIComponent(m.home)}${m.neutral ? '&neutral=1' : ''}${WEATHER[m.wx] ? `&wx=${m.wx}` : ''}`;

  window.WM = { loadLeague, buildCustom, recent, remember, url, WEATHER };
})();
