/* Team colors for matchup pages. Uses each team's official primary color (CFBD), keeping it
   readable on the white page and different enough from the opponent:
   1. a color too light to see on white (contrast < 3:1) is darkened until it reads, or the
      alternate is used when the primary is very light (e.g. gold) and the alternate is strong;
   2. if the two teams' colors look alike, the visiting team switches to its alternate color,
      then the home team does; failing that, the visitor falls back to the default blue/orange.
   Sets --home, --away, --home-fill and --away-fill on the page. Missing data keeps the defaults. */
(function () {
  const hex2rgb = h => [1, 3, 5].map(i => parseInt(h.slice(i, i + 2), 16));
  const rgb2hex = c => '#' + c.map(v => Math.round(Math.max(0, Math.min(255, v))).toString(16).padStart(2, '0')).join('');
  const lin = v => { v /= 255; return v <= 0.04045 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4; };
  const lum = c => 0.2126 * lin(c[0]) + 0.7152 * lin(c[1]) + 0.0722 * lin(c[2]);
  const contrastOnWhite = c => 1.05 / (lum(c) + 0.05);
  function lab(c) {
    const [r, g, b] = c.map(lin);
    let x = (r * 0.4124 + g * 0.3576 + b * 0.1805) / 0.95047, y = r * 0.2126 + g * 0.7152 + b * 0.0722, z = (r * 0.0193 + g * 0.1192 + b * 0.9505) / 1.08883;
    const f = t => t > 0.008856 ? Math.cbrt(t) : 7.787 * t + 16 / 116;
    [x, y, z] = [f(x), f(y), f(z)];
    return [116 * y - 16, 500 * (x - y), 200 * (y - z)];
  }
  const chroma = c => { const [, a, b] = lab(c); return Math.hypot(a, b); };
  const deltaE = (a, b) => { const p = lab(a), q = lab(b); return Math.hypot(p[0] - q[0], p[1] - q[1], p[2] - q[2]); };
  // Darken toward black until it has 3:1 contrast on white (a line/graphic minimum)
  function readable(h) {
    if (!h || !/^#[0-9a-f]{6}$/i.test(h)) return null;
    let c = hex2rgb(h);
    for (let i = 0; i < 20 && contrastOnWhite(c) < 3; i++) c = c.map(v => v * 0.88);
    return c;
  }
  // Near-white or near-gray colors make weak team colors; prefer the alternate when the primary is one
  const weak = h => { const c = hex2rgb(h); const [L, a, b] = lab(c); return L > 92 || Math.hypot(a, b) < 6 && L > 80; };
  const DEF = { away: '#2952a3', home: '#c2410c' };
  // Two colors "look alike" if close overall, the same hue family, or both near-black
  function alike(x, y) {
    const p = lab(x), q = lab(y);
    if (Math.hypot(p[0] - q[0], p[1] - q[1], p[2] - q[2]) < 40) return true;
    const cp0 = Math.hypot(p[1], p[2]), cq0 = Math.hypot(q[1], q[2]);
    if (p[0] < 28 && q[0] < 28 && (cp0 < 20 || cq0 < 20)) return true;   // black vs a very dark color
    const cp = Math.hypot(p[1], p[2]), cq = Math.hypot(q[1], q[2]);
    if (cp > 20 && cq > 20) {
      let dh = Math.abs(Math.atan2(p[2], p[1]) - Math.atan2(q[2], q[1])) * 180 / Math.PI;
      if (dh > 180) dh = 360 - dh;
      if (dh < 40) return true;
    }
    return false;
  }

  function pick(map, home, away) {
    const ok = h => h && /^#[0-9a-f]{6}$/i.test(h) && !weak(h);
    const opts = t => { let [p, a] = (map && map[t]) || [];
      // A very light primary (Iowa's gold) would need heavy darkening; use a strong alternate instead
      if (ok(p) && ok(a) && contrastOnWhite(hex2rgb(p)) < 2 && contrastOnWhite(hex2rgb(a)) >= 3) [p, a] = [a, p];
      // CFBD lists black as some teams' primary (Cincinnati: black, alt red). If the alternate is a
      // strong color that reads on white as-is, use it; black-and-light-gold teams (Iowa) stay black.
      else if (ok(p) && ok(a) && chroma(hex2rgb(p)) < 10 && chroma(hex2rgb(a)) > 30 && contrastOnWhite(hex2rgb(a)) >= 3) [p, a] = [a, p];
      return [p, a].filter(ok).map(readable); };
    const H = opts(home), A = opts(away);
    let h = H[0] || hex2rgb(DEF.home), a = A[0] || hex2rgb(DEF.away);
    if (alike(h, a)) {
      const tries = [[h, A[1]], [H[1], a], [H[1], A[1]]].filter(([x, y]) => x && y);
      const good = tries.find(([x, y]) => !alike(x, y));
      if (good) [h, a] = good;
      else a = [hex2rgb(DEF.away), hex2rgb(DEF.home), [55, 65, 81]].find(c => !alike(h, c)) || hex2rgb(DEF.away);
    }
    return { home: rgb2hex(h), away: rgb2hex(a) };
  }
  function apply(home, away, map) {
    const c = pick(map, home, away), st = document.documentElement.style;
    for (const side of ['home', 'away']) {
      const [r, g, b] = hex2rgb(c[side]);
      st.setProperty(`--${side}`, c[side]);
      st.setProperty(`--${side}-fill`, `rgba(${r}, ${g}, ${b}, .22)`);
    }
    return c;
  }
  async function load() {
    try { const r = await fetch('data/colors.json', { cache: 'no-cache' }); return r.ok ? await r.json() : null; } catch (e) { return null; }
  }
  // One team's readable color on its own (e.g. a swatch in a list)
  function single(map, team) {
    const [p, a] = (map && map[team]) || [];
    const ok = h => h && /^#[0-9a-f]{6}$/i.test(h) && !weak(h);
    let h = ok(p) ? p : ok(a) ? a : null;
    if (ok(p) && ok(a) && (contrastOnWhite(hex2rgb(p)) < 2 && contrastOnWhite(hex2rgb(a)) >= 3
        || chroma(hex2rgb(p)) < 10 && chroma(hex2rgb(a)) > 30 && contrastOnWhite(hex2rgb(a)) >= 3)) h = a;
    return h ? rgb2hex(readable(h)) : '#9ca3af';
  }
  window.TeamColors = { load, pick, apply, single };
})();
