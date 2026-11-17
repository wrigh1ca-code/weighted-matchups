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

  function render() {
    const term = q.value.trim().toLowerCase();
    list.textContent = '';
    data.games
      .filter(g => !term || g.home.toLowerCase().includes(term) || g.away.toLowerCase().includes(term))
      .forEach(g => {
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
        a.querySelector('.small').textContent = `${when} · ${g.away} ${at} ${g.home}`;
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
  render();
})();
