// Shared helpers for the three layout mockups. Data: data.js (a real game: steps = one model decision each).
const S = window.STEPS;
const esc = (s) => String(s ?? '').replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const pct = (x) => Math.round(x * 100) + '%';
const sid = (n) => { const [b, ...r] = n.toLowerCase().split('-'); return [b, r.join('')].filter(Boolean).join('-').replace(/[^a-z0-9-]/g, ''); };
const SPR = 'https://play.pokemonshowdown.com/sprites/';
const sprite = (n, back) => `<img class="spr" data-id="${sid(n)}" data-back="${back ? 1 : 0}" src="${SPR}${back ? 'ani-back' : 'ani'}/${sid(n)}.gif" alt="${esc(n)}" onerror="spriteFallback(this)">`;
// animated sprite -> static sprite -> base species (new Megas have no sprite yet) -> hidden
function spriteFallback(img) {
  const step = Number(img.dataset.f || 0); img.dataset.f = step + 1;
  const back = img.dataset.back === '1', id = img.dataset.id;
  if (step === 0) img.src = `${SPR}${back ? 'gen5-back' : 'gen5'}/${id}.png`;
  else if (step === 1) img.src = `${SPR}${back ? 'ani-back' : 'ani'}/${id.split('-')[0]}.gif`;
  else img.style.visibility = 'hidden';
}
const icon = (n) => `<img class="ico" src="${SPR}gen5/${sid(n)}.png" alt="${esc(n)}" title="${esc(n)}" onerror="this.style.visibility='hidden'">`;
const hpColor = (h) => (h > 0.5 ? 'var(--good)' : h > 0.2 ? 'var(--warn)' : 'var(--bad)');
const actives = (side) => [0, 1].map((i) => side.find((m) => m.active === i));
const stepName = (s) => `Turn ${s.turn}${s.forced ? ' · replacement' : ''}`;
const split = (label) => (label || '').split(' → ');

// What the model picked for each slot, as labels from the legal-move menu.
function picks(step) {
  return step.choice.split(', ').map((c, k) => {
    const o = step.menu[k] && step.menu[k].options.find((x) => x.choice === c.replace(' mega', ''));
    return o ? o.label : null;
  });
}
// Per slot: moves with probability summed over targets, then switches.
function moveGroups(t) {
  const g = {};
  for (const o of t.options) {
    const [m, target] = split(o.label);
    const row = (g[m] ??= { name: m, kind: o.kind, prob: 0, targets: [] });
    row.prob += o.prob;
    if (target) row.targets.push({ foe: target, prob: o.prob, damage: o.damage });
  }
  return Object.values(g).sort((a, b) => b.prob - a.prob);
}
// Cumulative battle log up to and including step i.
const logUpTo = (i) => S.slice(0, i + 1).flatMap((s) => s.events);
const eventLine = (e) => `<li class="${esc(e.tone)}">${esc(e.text)}</li>`;
const winSeries = () => S.map((s) => (s.thoughts[0] ? s.thoughts[0].winChance : null));

// Payoff matrix: our candidate plans (rows) against their likely replies (columns), one-turn health margin.
function matrix(step, { rows = 6, short = false } = {}) {
  const a = step.analysis;
  if (!a || !a.rows.length) return '<p class="quiet">No search this turn (a forced replacement has one sensible set of options).</p>';
  const heat = (x) => (x == null ? 'transparent' : x >= 0 ? `rgba(46,160,90,${Math.min(0.55, x * 1.6)})` : `rgba(214,66,60,${Math.min(0.55, -x * 1.6)})`);
  const sg = (x) => (x == null ? '–' : (x >= 0 ? '+' : '') + x.toFixed(2));
  const rs = [...a.rows].sort((x, y) => (y.score ?? -9) - (x.score ?? -9)).slice(0, rows);
  const head = a.replies.map((r, i) => `<th title="${esc(r.label)}">${short ? 'R' + (i + 1) : esc(r.label.split(' + ').map((p) => split(p)[0]).join(' + '))}</th>`).join('');
  const body = rs.map((r) => `<tr class="${r.choice === a.choice ? 'pick' : ''}"><td class="plan">${esc(r.label)}</td>${r.payoffs.map((x) => `<td style="background:${heat(x)}">${sg(x)}</td>`).join('')}<td class="num">${sg(r.mean)}</td><td class="num">${sg(r.worst)}</td><td class="num strong">${sg(r.score)}</td></tr>`).join('');
  return `<div class="mx"><table><thead><tr><th>Our plan</th>${head}<th>mean</th><th>worst</th><th>score</th></tr></thead><tbody>${body}</tbody></table></div>`;
}

// Wire the arrow keys and prev/next buttons; render(i) draws the whole page for step i.
function mount(render, start = 0) {
  let i = start;
  const go = (d) => { i = Math.max(0, Math.min(S.length - 1, typeof d === 'number' ? i + d : d.set)); render(i); };
  window.addEventListener('keydown', (e) => { if (e.key === 'ArrowLeft') go(-1); if (e.key === 'ArrowRight') go(1); });
  document.addEventListener('click', (e) => {
    const b = e.target.closest('[data-step]');
    if (!b) return;
    const v = b.dataset.step;
    go(v === 'prev' ? -1 : v === 'next' ? 1 : { set: Number(v) });
  });
  render(i);
}
