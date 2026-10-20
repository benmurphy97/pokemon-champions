// The battle dashboard: one page that shows everything about a decision. Used by the match viewer (watch) and the
// play mode. A "step" is one decision of the model (see match_viewer.look): board, thoughts, analysis, events.
import { useState } from 'react'
import Sprite from './Sprite.jsx'
import './battle.css'

const pct = (x) => `${Math.round(x * 100)}%`
const base = (n) => n.split('-')[0]
const same = (a, b) => base(a) === base(b)
const split = (label) => (label || '').split(' → ')
const STATS = [['hp', 'HP'], ['atk', 'Atk'], ['def', 'Def'], ['spa', 'SpA'], ['spd', 'SpD'], ['spe', 'Spe']]
const hpColor = (h) => (h > 0.5 ? '#1f9d57' : h > 0.2 ? '#d9930d' : '#d23f3a')
const actives = (side) => [0, 1].map((i) => side.find((m) => m.active === i))

// What the model picked for each slot: from the menu in play mode, else from the search row, else the policy's choice.
export function picks(step) {
  if (step.menu) {
    return step.choice.split(', ').map((c, k) => step.menu[k]?.options.find((o) => o.choice === c.replace(' mega', ''))?.label ?? null)
  }
  const row = step.analysis?.rows.find((r) => r.choice === step.analysis.choice)
  if (row) return row.label.split(' + ')
  return step.thoughts.map((t) => t.chosen)
}
export function moveGroups(thought) {
  const groups = {}
  for (const o of thought.options) {
    const [name, target] = split(o.label)
    const g = (groups[name] ??= { name, kind: o.kind, prob: 0, targets: [] })
    g.prob += o.prob
    if (target) g.targets.push({ foe: target, prob: o.prob, damage: o.damage })
  }
  return Object.values(groups).sort((a, b) => b.prob - a.prob)
}
const ourSet = (step, m) => step.board.oursSets?.find((s) => same(s.name, m.name))
function ourDecision(step, m) {
  const t = m.active != null ? step.thoughts[m.active] : null
  if (!t || !t.actor) return null
  const chosen = split(picks(step)[m.active] || t.chosen)[0]
  return { groups: moveGroups(t), chosen: chosen.replace(/^switch to /i, 'Switch to ') }
}
const foePred = (step, m) => step.thoughts.find((t) => t.foePrediction?.length)?.foePrediction.find((f) => same(f.foe, m.name)) ?? null
function switchIn(step, m) {
  let p = null
  for (const t of step.thoughts) for (const o of t.options) if (o.label === `Switch to ${m.name}` || o.label === `Switch to ${base(m.name)}`) p = Math.max(p ?? 0, o.prob)
  return p
}
export const winOf = (step) => step.thoughts[0]?.winChance ?? 0.5

const Hp = ({ value }) => <div className="hp"><i style={{ width: `${value * 100}%`, background: hpColor(value) }} /></div>
const Stats = ({ stats, estimated }) => (
  <div className={`stats ${estimated ? 'est' : ''}`}>{STATS.map(([k, label]) => <span key={k}><i>{label}</i><b>{stats[k] ?? '–'}</b></span>)}</div>
)
const Fact = ({ f }) => <span className={`fact ${f.known ? 'known' : 'guess'}`}>{f.name} <b>{f.known ? '✓ 100%' : pct(f.p)}</b></span>

function OurCard({ step, m }) {
  const set = ourSet(step, m) ?? { moves: [], stats: {} }
  const dec = ourDecision(step, m)
  const sw = switchIn(step, m)
  const hp = set.stats?.hp ? `${Math.round(m.hp * set.stats.hp)}/${set.stats.hp}` : pct(m.hp)
  const state = m.fainted ? 'fainted' : m.active != null ? 'on the field' : 'bench'
  return (
    <div className={`unit us ${m.fainted ? 'out' : m.active != null ? 'live' : ''}`}>
      <div className="uh"><Sprite name={m.name} /><div><b>{m.name}</b><span className="sub">{state}{sw != null && !m.fainted ? ` · switch-in ${pct(sw)}` : ''}</span></div><span className="hpn">{m.fainted ? '0' : hp}</span></div>
      <Hp value={m.hp} />
      <div className="sub">{[set.item, set.ability].filter(Boolean).join(' · ')}</div>
      {set.moves.map((name) => {
        const g = dec?.groups.find((x) => x.name === name)
        const chosen = dec && dec.chosen === name
        return (
          <div key={name} className={`mv ${chosen ? 'pick' : ''}`}>
            <span>{chosen ? '▶ ' : ''}{name}</span>
            {dec ? <><span className="bar"><i style={{ width: `${(g?.prob ?? 0) * 100}%` }} /></span><b>{pct(g?.prob ?? 0)}</b></> : <><span /><b /></>}
          </div>
        )
      })}
      {dec && <div className="chips">{dec.groups.filter((g) => g.kind === 'switch').map((g) => <span key={g.name} className="chip">{g.name} {pct(g.prob)}</span>)}</div>}
      {set.stats?.atk && <Stats stats={set.stats} />}
    </div>
  )
}

function FoeCard({ step, m, index, rows = 5 }) {
  const belief = step.board.foeSets?.[index]
  if (!belief) return null
  const pred = foePred(step, m)
  const state = m.fainted ? 'fainted' : m.active != null ? 'on the field' : m.seen ? 'revealed' : 'not seen yet'
  const stats = belief.expectedStats ?? belief.spread?.stats
  return (
    <div className={`unit foe ${m.seen ? 'seen' : 'unseen'} ${m.fainted ? 'out' : m.active != null ? 'live' : ''}`}>
      <div className="uh"><Sprite name={m.name} /><div><b>{m.name}</b><span className="sub">{state}</span></div><span className="hpn">{m.seen ? pct(m.hp) : ''}</span></div>
      {m.seen && <Hp value={m.hp} />}
      {[['Item', belief.items], ['Ability', belief.abilities]].map(([k, xs]) => (
        <div className="facts" key={k}><span className="k">{k}</span>{xs.length ? xs.map((f) => <Fact key={f.name} f={f} />) : <span className="sub">unknown</span>}</div>
      ))}
      {pred && <div className="legend2"><span>has it</span><span>uses it now</span></div>}
      {belief.moves.slice(0, rows).map((mv) => {
        const use = pred?.top.find((x) => x.label === mv.name)
        return (
          <div key={mv.name} className={`mv foe ${use && pred.top[0] === use ? 'pick' : ''}`}>
            <span>{mv.name}</span>
            <span className="has"><i className={mv.known ? 'k' : ''} style={{ width: `${mv.p * 100}%` }} /></span>
            <b className={mv.known ? 'k' : ''}>{mv.known ? '✓' : pct(mv.p)}</b>
            {pred && <b className="use">{use ? pct(use.prob) : ''}</b>}
          </div>
        )
      })}
      {stats && <><Stats stats={stats} estimated />
        <div className="sub">estimated{belief.spread ? `: ${belief.spread.nature}, ${pct(belief.spread.p)} of matching sets` : ''}{belief.evidence != null ? ` · ~${Math.max(1, Math.round(belief.evidence))} matching sets` : ''}</div></>}
    </div>
  )
}

function Stage({ step }) {
  const b = step.board
  const foe = actives(b.foe), us = actives(b.ours)
  const pos = { foe: [[30, 46], [70, 52]], us: [[22, 94], [58, 90]] }
  const one = (m, p, back) => m && (
    <div key={`${back}${m.name}`} className="sm" style={{ left: `${p[0]}%`, top: `${p[1]}%` }}>
      <Sprite name={m.name} animated back={back} />
      <div className="plate2"><b>{m.name}</b> <i>{pct(m.hp)}</i><Hp value={m.hp} />{m.status && <span className="chip">{m.status}</span>}
        {Object.entries(m.boosts || {}).map(([k, v]) => <span key={k} className="chip up">{k} {v > 0 ? '+' : ''}{v}</span>)}</div>
    </div>
  )
  return (
    <div className={`stage2 ${b.weather === 'sun' ? 'sun' : ''}`}>
      <div className="tags2"><span className="tag">Turn {step.turn}{step.forced ? ' · replacement' : ''}</span>
        {b.weather && <span className="tag">{b.weather}</span>}{b.terrain && <span className="tag">{b.terrain}</span>}{b.trickRoom && <span className="tag">Trick Room</span>}
        {(b.foeConditions || []).map((c) => <span key={`f${c}`} className="tag">foe {c}</span>)}{(b.ourConditions || []).map((c) => <span key={`o${c}`} className="tag">ours {c}</span>)}</div>
      {foe.map((m, k) => one(m, pos.foe[k], false))}{us.map((m, k) => one(m, pos.us[k], true))}
    </div>
  )
}

const signed = (x) => (x == null ? '–' : `${x >= 0 ? '+' : ''}${x.toFixed(2)}`)
const heat = (x) => (x == null ? 'transparent' : x >= 0 ? `rgba(46,160,90,${Math.min(0.55, x * 1.6)})` : `rgba(214,66,60,${Math.min(0.55, -x * 1.6)})`)

function Lookahead({ step }) {
  const a = step.analysis
  if (!a || !a.rows.length) return <p className="quiet">{step.forced ? 'Replacement: one sensible set of options, so no look-ahead.' : 'No search this turn.'}</p>
  const rows = [...a.rows].sort((x, y) => (y.score ?? -9) - (x.score ?? -9)).slice(0, 6)
  return (
    <>
      <h3><span>Candidate plans</span><span>policy vs one-turn search</span></h3>
      {rows.slice(0, 5).map((r, j) => (
        <div key={r.choice} className={`cand ${r.choice === a.choice ? 'pick' : ''}`}>
          <span className="n">{j + 1}</span><div>{r.label}<div className="bar"><i style={{ width: `${r.prior * 100}%` }} /></div></div>
          <b>{pct(r.prior)}<small>policy</small></b><b>{signed(r.score)}<small>search</small></b>
        </div>
      ))}
      <h3 style={{ marginTop: 10 }}><span>Payoff matrix</span><span>health margin after 1 turn, ours − theirs{a.samples > 1 ? `, averaged over ${a.samples} sampled opponent sets` : ''}</span></h3>
      <div className="mx"><table>
        <thead><tr><th>Our plan</th>{a.replies.map((r, i) => <th key={i} title={r.label}>{r.label.split(' + ').map((p) => split(p)[0]).join(' + ')}</th>)}<th>mean</th><th>worst</th><th>score</th></tr></thead>
        <tbody>{rows.map((r) => (
          <tr key={r.choice} className={r.choice === a.choice ? 'pick' : ''}>
            <td className="plan">{r.label}</td>
            {r.payoffs.map((x, i) => <td key={i} style={{ background: heat(x) }}>{signed(x)}</td>)}
            <td>{signed(r.mean)}</td><td>{signed(r.worst)}</td><td className="strong">{signed(r.score)}</td>
          </tr>))}</tbody>
      </table></div>
    </>
  )
}

export function Log({ steps, upTo }) {
  return <ul className="log">{steps.slice(0, upTo + 1).flatMap((s) => s.events).map((e, i) => <li key={i} className={e.tone}>{e.text}</li>)}</ul>
}

// steps: every decision so far; index: the one shown; children: extra controls (play mode: your move) under the field.
export default function Battle({ steps, index, onIndex, children }) {
  const [tab, setTab] = useState('look')
  const step = steps[index]
  const b = step.board
  const ours = [...b.ours].sort((x, y) => (y.active != null) - (x.active != null) || x.fainted - y.fainted)
  const foes = b.foe.map((m, k) => ({ m, k })).sort((x, y) => (y.m.active != null) - (x.m.active != null) || y.m.seen - x.m.seen || x.m.fainted - y.m.fainted)
  const w = winOf(step)
  return (
    <div className="bd">
      <div className="col"><h3><span>Your four</span><span>bars: policy odds · ▶ final pick</span></h3>{ours.map((m) => <OurCard key={m.name} step={step} m={m} />)}</div>
      <div className="mid">
        <div className="bar2">
          <div className="stepper2">
            <button onClick={() => onIndex(Math.max(0, index - 1))} disabled={index === 0}>‹</button>
            {steps.map((s, j) => <button key={j} className={j === index ? 'on' : ''} onClick={() => onIndex(j)}>{s.forced ? '↳' : s.turn}</button>)}
            <button onClick={() => onIndex(Math.min(steps.length - 1, index + 1))} disabled={index === steps.length - 1}>›</button>
          </div>
          <div className="winbar"><span>You {pct(w)}</span><div><i style={{ width: `${w * 100}%` }} /></div><span>{pct(1 - w)} Them</span></div>
        </div>
        <Stage step={step} />
        {children}
        <div className="card scroll">
          <div className="tabs2">{[['look', '1-turn look-ahead'], ['log', 'Battle log']].map(([k, n]) => <button key={k} className={tab === k ? 'on' : ''} onClick={() => setTab(k)}>{n}</button>)}</div>
          {tab === 'look' ? <Lookahead step={step} /> : <Log steps={steps} upTo={index} />}
        </div>
      </div>
      <div className="col"><h3><span>Their six</span><span>solid = revealed</span></h3>{foes.map(({ m, k }) => <FoeCard key={k} step={step} m={m} index={k} />)}</div>
    </div>
  )
}
