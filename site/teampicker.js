/* Team picker: a combobox you can type into (filters as you type) or open as a full dropdown
   grouped by conference. Keyboard: ↓/↑ move, Enter picks, Esc closes, Tab picks the highlighted
   or only match. Follows the WAI-ARIA combobox pattern.

   const p = TeamPicker(rootEl, teams, { placeholder, onChange });  teams: [{team, conference, games}]
   p.value -> selected team name or ''     p.set(name)     p.focus() */
(function () {
  const norm = s => String(s).toLowerCase().normalize('NFD').replace(/[̀-ͯ]/g, '').replace(/[^a-z0-9 ]/g, '');
  const esc = s => String(s).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  let uid = 0;

  window.TeamPicker = function (root, teams, opts = {}) {
    const id = `tp${++uid}`;
    const sorted = [...teams].sort((a, b) => a.team.localeCompare(b.team));
    const confs = [...new Set(sorted.map(t => t.conference || 'Other'))].sort();
    const inputId = root.dataset.inputId || `${id}-input`;
    root.classList.add('combo');
    root.innerHTML = `
      <input id="${inputId}" type="text" role="combobox" aria-autocomplete="list" aria-expanded="false"
        aria-controls="${id}-list" autocomplete="off" spellcheck="false" placeholder="${esc(opts.placeholder || 'Type or choose a team')}">
      <button type="button" class="combo-toggle" tabindex="-1" aria-label="Show all teams" aria-controls="${id}-list">
        <svg width="14" height="14" viewBox="0 0 24 24" aria-hidden="true"><path d="M6 9l6 6 6-6" style="fill:none;stroke:currentColor;stroke-width:2.5;stroke-linecap:round;stroke-linejoin:round"/></svg></button>
      <ul class="combo-list" id="${id}-list" role="listbox" hidden></ul>`;
    const input = root.querySelector('input'), list = root.querySelector('ul'), toggle = root.querySelector('button');
    let value = '', shown = [], active = -1;

    function matches(term) {
      const q = norm(term).trim();
      if (!q) return null;
      const scored = [];
      for (const t of sorted) {
        const n = norm(t.team), c = norm(t.conference || '');
        let s = -1;
        if (n === q) s = 0;
        else if (n.startsWith(q)) s = 1;
        else if (n.split(' ').some(w => w.startsWith(q))) s = 2;
        else if (n.includes(q)) s = 3;
        else if (c.startsWith(q)) s = 4;
        if (s >= 0) scored.push([s, t]);
      }
      return scored.sort((a, b) => a[0] - b[0] || a[1].team.localeCompare(b[1].team)).map(x => x[1]);
    }

    function render(term) {
      const m = matches(term);
      shown = m || sorted.slice().sort((a, b) => (a.conference || 'Other').localeCompare(b.conference || 'Other') || a.team.localeCompare(b.team));
      let html = '', lastConf = null;
      if (!shown.length) html = '<li class="combo-empty" role="presentation">No FBS team matches</li>';
      shown.forEach((t, i) => {
        const conf = t.conference || 'Other';
        if (!m && conf !== lastConf) { html += `<li class="combo-group" role="presentation">${esc(conf)}</li>`; lastConf = conf; }
        html += `<li role="option" id="${id}-o${i}" data-i="${i}" aria-selected="${t.team === value}"${t.team === value ? ' class="sel"' : ''}>
          <span>${esc(t.team)}</span>${m ? `<small>${esc(conf)}</small>` : ''}</li>`;
      });
      list.innerHTML = html;
      active = m && shown.length ? 0 : shown.findIndex(t => t.team === value);
      highlight();
    }
    function highlight() {
      list.querySelectorAll('[role=option]').forEach(li => li.classList.toggle('active', +li.dataset.i === active));
      if (active >= 0) {
        input.setAttribute('aria-activedescendant', `${id}-o${active}`);
        const li = list.querySelector(`#${id}-o${active}`);
        if (li) li.scrollIntoView({ block: 'nearest' });
      } else input.removeAttribute('aria-activedescendant');
    }
    function open(term) { render(term); list.hidden = false; input.setAttribute('aria-expanded', 'true'); }
    function close() { list.hidden = true; input.setAttribute('aria-expanded', 'false'); input.removeAttribute('aria-activedescendant'); }
    function set(name, fire = true) {
      const t = sorted.find(x => x.team === name);
      value = t ? t.team : '';
      input.value = value;
      root.classList.toggle('picked', !!value);
      if (fire && opts.onChange) opts.onChange(value);
    }
    function pick(i) { if (shown[i]) set(shown[i].team); close(); }
    // Leaving the field: accept an exact (case/punctuation-insensitive) name, otherwise keep the last pick
    function commitTyped() {
      const q = norm(input.value).trim();
      if (!q) { if (value) set(''); return; }
      const exact = sorted.find(t => norm(t.team) === q);
      if (exact) { if (exact.team !== value) set(exact.team); else input.value = value; }
      else if (value && norm(value) !== q) { const m = matches(input.value); if (m && m.length === 1) set(m[0].team); else { input.value = value; } }
      else if (!value) { const m = matches(input.value); if (m && m.length === 1) set(m[0].team); }
    }

    input.addEventListener('input', () => {
      if (value && input.value !== value) { value = ''; root.classList.remove('picked'); if (opts.onChange) opts.onChange(''); }
      open(input.value);
    });
    input.addEventListener('focus', () => { if (input.value) input.select(); });
    input.addEventListener('click', () => { if (list.hidden) open(value ? '' : input.value); });
    input.addEventListener('keydown', e => {
      const n = shown.length;
      if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
        e.preventDefault();
        if (list.hidden) { open(value ? '' : input.value); return; }
        if (!n) return;
        active = e.key === 'ArrowDown' ? (active + 1) % n : (active <= 0 ? n - 1 : active - 1);
        highlight();
      } else if (e.key === 'Enter') {
        if (!list.hidden && active >= 0) { e.preventDefault(); pick(active); }
        else if (!value) { e.preventDefault(); commitTyped(); }
      } else if (e.key === 'Escape') {
        if (!list.hidden) { e.preventDefault(); close(); input.value = value; }
      } else if (e.key === 'Tab') {
        if (!list.hidden && active >= 0 && input.value && input.value !== value) pick(active);
        close();
      }
    });
    input.addEventListener('blur', () => setTimeout(() => {
      if (root.contains(document.activeElement)) return;
      close(); commitTyped();
    }, 120));
    list.addEventListener('mousedown', e => e.preventDefault());   // keep focus in the input
    list.addEventListener('click', e => { const li = e.target.closest('[role=option]'); if (li) { pick(+li.dataset.i); input.focus(); } });
    list.addEventListener('mousemove', e => { const li = e.target.closest('[role=option]'); if (li && +li.dataset.i !== active) { active = +li.dataset.i; highlight(); } });
    toggle.addEventListener('mousedown', e => e.preventDefault());
    toggle.addEventListener('click', () => { if (list.hidden) { open(''); input.focus(); } else close(); });
    document.addEventListener('mousedown', e => { if (!root.contains(e.target)) close(); });

    return { get value() { return value; }, set: (n) => set(n), focus: () => input.focus(), input };
  };
})();
