// Helpers for the second round of layouts: our four, their six (with beliefs), the 2v2 field and the decision data.
const base = (n) => n.split('-')[0];
const same = (a, b) => base(a) === base(b);
const LAB = { hp: 'HP', atk: 'Atk', def: 'Def', spa: 'SpA', spd: 'SpD', spe: 'Spe' };
const hpBar = (h) => `<div class="hp"><i style="width:${h * 100}%;background:${hpColor(h)}"></i></div>`;
const ourSet = (step, m) => step.board.oursSets.find((s) => same(s.name, m.name));
function ourDecision(step, m) {
  const t = m.active != null ? step.thoughts[m.active] : null;
  return t && t.actor ? { t, groups: moveGroups(t), chosen: split(picks(step)[m.active] || t.chosen)[0] } : null;
}
function foePred(step, m) {
  const t = step.thoughts.find((x) => x.foePrediction && x.foePrediction.length);
  return t ? (t.foePrediction.find((f) => same(f.foe, m.name)) || null) : null;
}
// Chance the model brings this benched Pokémon in (largest "Switch to X" or replacement option).
function switchIn(step, m) {
  let p = null;
  for (const t of step.thoughts) for (const o of t.options) if (o.label === `Switch to ${m.name}` || o.label === `Switch to ${base(m.name)}`) p = Math.max(p ?? 0, o.prob);
  return p;
}
const statStrip = (st, est) => `<div class="stats ${est ? 'est' : ''}">${['hp', 'atk', 'def', 'spa', 'spd', 'spe'].map((k) => `<span><i>${LAB[k]}</i><b>${st[k] ?? '–'}</b></span>`).join('')}</div>`;
const fact = (f) => `<span class="fact ${f.known ? 'known' : 'guess'}">${esc(f.name)} <b>${f.known ? '✓ 100%' : pct(f.p)}</b></span>`;

// Our Pokémon: exact set, the model's odds per move when it is on the field.
function ourCard(step, m, { stats = true } = {}) {
  const set = ourSet(step, m) || { moves: [], stats: {} }, dec = ourDecision(step, m), sw = switchIn(step, m);
  const cur = set.stats.hp ? `${Math.round(m.hp * set.stats.hp)}/${set.stats.hp}` : pct(m.hp);
  const moves = set.moves.map((name) => {
    const g = dec && dec.groups.find((x) => x.name === name), p = g ? g.prob : 0, chosen = dec && dec.chosen === name;
    return `<div class="mv ${chosen ? 'pick' : ''}"><span>${chosen ? '▶ ' : ''}${esc(name)}</span>${dec ? `<span class="bar"><i style="width:${p * 100}%"></i></span><b>${pct(p)}</b>` : '<span></span><b></b>'}</div>`;
  }).join('');
  const switches = dec ? dec.groups.filter((g) => g.kind === 'switch').map((g) => `<span class="chip">${esc(g.name)} ${pct(g.prob)}</span>`).join('') : '';
  return `<div class="card unit us ${m.fainted ? 'out' : m.active != null ? 'live' : ''}" data-pick="us|${esc(m.name)}">
   <div class="uh">${icon(m.name)}<div><b>${esc(m.name)}</b><span class="sub">${m.fainted ? 'fainted' : m.active != null ? 'on the field' : 'bench'}${sw != null && !m.fainted ? ` · switch-in ${pct(sw)}` : ''}</span></div><span class="hpn">${m.fainted ? '0' : cur}</span></div>${hpBar(m.hp)}
   <div class="sub">${[set.item, set.ability].filter(Boolean).map(esc).join(' · ')}</div>${moves}${switches ? `<div class="chips">${switches}</div>` : ''}${stats && set.stats.atk ? statStrip(set.stats) : ''}</div>`;
}
// Their Pokémon: what we know and what we guess, each fact with its probability; ✓ 100% when revealed.
function foeCard(step, m, i, { moves = 8, stats = true } = {}) {
  const bl = step.board.foeSets[i], pred = foePred(step, m);
  const rows = bl.moves.slice(0, moves).map((mv) => {
    const use = pred && pred.top.find((x) => x.label === mv.name);
    return `<div class="mv ${use && pred.top[0] === use ? 'pick' : ''}"><span>${esc(mv.name)}</span><span class="has" title="chance they carry it"><i style="width:${mv.p * 100}%" class="${mv.known ? 'k' : ''}"></i></span><b class="${mv.known ? 'k' : ''}">${mv.known ? '✓' : pct(mv.p)}</b>${pred ? `<b class="use" title="chance they use it this turn">${use ? pct(use.prob) : ''}</b>` : ''}</div>`;
  }).join('');
  const facts = [['Item', bl.items], ['Ability', bl.abilities]].map(([k, xs]) => `<div class="facts"><span class="k">${k}</span>${xs.length ? xs.map(fact).join('') : '<span class="sub">unknown</span>'}</div>`).join('');
  const sp = bl.spread;
  return `<div class="card unit foe ${m.seen ? 'seen' : 'unseen'} ${m.fainted ? 'out' : m.active != null ? 'live' : ''}" data-pick="foe|${i}">
   <div class="uh">${icon(m.name)}<div><b>${esc(m.name)}</b><span class="sub">${m.fainted ? 'fainted' : m.active != null ? 'on the field' : m.seen ? 'revealed' : 'not seen yet'}</span></div><span class="hpn">${m.seen ? pct(m.hp) : ''}</span></div>${m.seen ? hpBar(m.hp) : ''}
   ${facts}${pred ? '<div class="legend2"><span>has it</span><span>uses it now</span></div>' : ''}${rows}
   ${stats && sp ? `${statStrip(sp.stats, true)}<div class="sub">estimated: ${esc(sp.nature)} · ${pct(sp.p)} of ladder players run this spread</div>` : ''}</div>`;
}
// A small 2v2 battlefield.
function stage(step, { height = 230 } = {}) {
  const b = step.board, fo = actives(b.foe), us = actives(b.ours);
  const pos = { foe: [[34, 46], [66, 52]], us: [[22, 94], [56, 90]] };
  const one = (m, p, back) => m ? `<div class="sm" style="left:${p[0]}%;top:${p[1]}%">${sprite(m.name, back)}<div class="plate2"><b>${esc(m.name)}</b> <i>${pct(m.hp)}</i>${hpBar(m.hp)}</div></div>` : '';
  const w = b.weather ? `<span class="tag">${esc(b.weather)}</span>` : '';
  return `<div class="stage2 ${b.weather === 'sun' ? 'sun' : ''}" style="height:${height}px"><div class="tags2"><span class="tag">${stepName(step)}</span>${w}${b.trickRoom ? '<span class="tag">Trick Room</span>' : ''}${(b.foeConditions || []).map((c) => `<span class="tag">foe ${esc(c)}</span>`).join('')}${(b.ourConditions || []).map((c) => `<span class="tag">ours ${esc(c)}</span>`).join('')}</div>
   ${fo.map((m, k) => one(m, pos.foe[k], false)).join('')}${us.map((m, k) => one(m, pos.us[k], true)).join('')}</div>`;
}
// Candidate plans: policy prior next to what the one-turn search made of them.
function candidates(step, n = 5) {
  const a = step.analysis;
  if (!a || !a.rows.length) return `<p class="quiet">Replacement: ${esc(step.thoughts[0] ? step.thoughts[0].chosen : '')}. No look-ahead, there is one sensible set of options.</p>`;
  const rows = [...a.rows].sort((x, y) => (y.score ?? -9) - (x.score ?? -9)).slice(0, n);
  return rows.map((r, j) => `<div class="cand ${r.choice === a.choice ? 'pick' : ''}"><span class="n">${j + 1}</span><div>${esc(r.label)}<div class="bar"><i style="width:${r.prior * 100}%"></i></div></div><b>${pct(r.prior)}<small>policy</small></b><b>${r.score == null ? '–' : (r.score >= 0 ? '+' : '') + r.score.toFixed(2)}<small>search</small></b></div>`).join('');
}
const tabs = (cur, items) => `<div class="tabs2">${items.map(([k, n]) => `<button data-tab="${k}" class="${cur === k ? 'on' : ''}">${n}</button>`).join('')}</div>`;
const stepper = (i) => `<div class="stepper2"><button data-step="prev">‹</button>${S.map((x, j) => `<button data-step="${j}" class="${j === i ? 'on' : ''}">${x.forced ? '↳' : x.turn}</button>`).join('')}<button data-step="next">›</button></div>`;
const winOf = (s) => (s.thoughts[0] ? s.thoughts[0].winChance : 0.5);
const winBar = (w) => `<div class="winbar"><span>You ${pct(w)}</span><div><i style="width:${w * 100}%"></i></div><span>${pct(1 - w)} Them</span></div>`;
