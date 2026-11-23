(async function () {
  const res = await fetch('data/week.json', { cache: 'no-cache' });
  const data = await res.json();
  const meta = data.meta;
  document.getElementById('weeklbl').textContent = meta.season_type === 'postseason' ? `${meta.season} · Postseason` : `${meta.season} · Week ${meta.week}`;
  document.getElementById('gen').textContent = 'Updated ' + new Date(meta.generated).toLocaleString();
  const list = document.getElementById('list');
  const q = document.getElementById('q');

  function fmtSpread(g) {
    if (g.spread_home == null) return 'Market: n/a';
    const s = g.spread_home;
    if (Math.abs(s) < 0.25) return 'Market: pick';
    return s < 0 ? `Market: ${g.home} ${s}` : `Market: ${g.away} −${s}`;
  }

  // Conference filter: pick one or more; a game shows if either team is in a picked conference.
  // Kept in the URL (?conf=Big 12,Big Ten) so a filtered view can be bookmarked or shared.
  const SHORT = { 'American Athletic': 'American', 'Mid-American': 'MAC', 'Conference USA': 'C-USA', 'FBS Independents': 'Independents' };
  const POWER = ['ACC', 'Big 12', 'Big Ten', 'SEC'];
  const short = c => SHORT[c] || c;
  const counts = {};
  for (const g of data.games) for (const c of new Set([g.home_conf, g.away_conf])) if (c) counts[c] = (counts[c] || 0) + 1;
  const confs = Object.keys(counts).sort((a, b) =>
    (POWER.includes(b) - POWER.includes(a)) || (a === 'FBS Independents') - (b === 'FBS Independents') || a.localeCompare(b));
  const params = new URLSearchParams(location.search);
  const picked = new Set((params.get('conf') || '').split(',').map(x => x.trim()).filter(c => counts[c]));
  const box = document.getElementById('confs');
  const btn = (label, title, on, click) => {
    const b = document.createElement('button');
    b.type = 'button'; b.className = 'chip conf' + (on ? ' on' : ''); b.setAttribute('aria-pressed', on);
    b.innerHTML = label; if (title) b.title = title; b.addEventListener('click', click); return b;
  };
  function drawConfs() {
    box.textContent = '';
    const powerOn = POWER.every(c => picked.has(c) || !counts[c]) && picked.size === POWER.filter(c => counts[c]).length;
    box.append(btn('All', 'Show every game', picked.size === 0, () => { picked.clear(); update(); }));
    if (POWER.some(c => counts[c])) box.append(btn('Power 4', 'ACC, Big 12, Big Ten and SEC', powerOn, () => {
      picked.clear(); if (!powerOn) POWER.forEach(c => counts[c] && picked.add(c)); update(); }));
    for (const c of confs) box.append(btn(`${short(c)} <span class="n">${counts[c]}</span>`, c, picked.has(c), () => {
      picked.has(c) ? picked.delete(c) : picked.add(c); update(); }));
    document.getElementById('confnote').textContent = picked.size ? '· includes nonconference games' : '';
  }
  function update() {
    const u = new URL(location.href);
    if (picked.size) u.searchParams.set('conf', [...picked].join(',')); else u.searchParams.delete('conf');
    history.replaceState(null, '', u);
    drawConfs(); render();
  }

  function render() {
    const term = q.value.trim().toLowerCase();
    list.textContent = '';
    const shown = data.games
      .filter(g => !picked.size || picked.has(g.home_conf) || picked.has(g.away_conf))
      .filter(g => !term || g.home.toLowerCase().includes(term) || g.away.toLowerCase().includes(term));
    document.getElementById('count').textContent = (picked.size || term)
      ? `${shown.length} of ${data.games.length} games` : `${data.games.length} games`;
    if (!shown.length) {
      list.innerHTML = '<div class="note">No games match. Try another conference or clear the search.</div>';
      return;
    }
    shown.forEach(g => {
        const a = document.createElement('a');
        a.className = 'gcard';
        a.href = `game.html?id=${g.id}`;
        const when = new Date(g.start).toLocaleString([], { weekday: 'short', month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit' });
        const at = g.neutral ? 'vs' : 'at';
        a.innerHTML = `
          <div class="small"></div>
          <div class="teams"><span class="away"></span><span class="pts a"></span></div>
          <div class="teams"><span class="home"></span><span class="pts h"></span></div>
          <div class="small spread"></div>`;
        const cf = g.home_conf === g.away_conf ? short(g.home_conf || '') : [g.away_conf, g.home_conf].filter(Boolean).map(short).join(' / ');
        a.querySelector('.small').textContent = `${when}${cf ? ' · ' + cf : ''}`;
        a.querySelector('.teams .away').textContent = g.away;
        a.querySelector('.pts.a').textContent = Math.round(g.away_pts);
        a.querySelector('.teams .home').textContent = g.home;
        a.querySelector('.pts.h').textContent = Math.round(g.home_pts);
        const wp = g.home_wp >= 0.5 ? `${g.home} ${Math.round(g.home_wp * 100)}%` : `${g.away} ${Math.round((1 - g.home_wp) * 100)}%`;
        a.querySelector('.spread').textContent = `Win prob: ${wp} · ${fmtSpread(g)}`;
        list.appendChild(a);
      });
  }
  q.addEventListener('input', render);
  drawConfs();
  render();
})();
