import { useState } from 'react'
import Battle from './Battle.jsx'
import { Setup, useOptions } from './GameSetup.jsx'
import Sprite from './Sprite.jsx'

const pct = (x) => `${Math.round(x * 100)}%`

// Your choice for each slot, then the choice string the simulator expects ("move 1 2 mega, switch 3").
function YourMove({ step, busy, onPlay }) {
  const menu = step.menu.slice(0, step.slots)
  const suggested = step.choice.split(', ').map((c) => c.replace(' mega', ''))
  const [picks, setPicks] = useState(menu.map((m, i) => (m.options.some((o) => o.choice === suggested[i]) ? suggested[i] : m.options[0]?.choice)))
  const [mega, setMega] = useState(step.choice.includes('mega') ? step.choice.split(', ').findIndex((c) => c.includes('mega')) : -1)

  const sameSwitch = picks.length === 2 && picks[0] === picks[1] && picks[0].startsWith('switch')
  const choice = picks.map((p, i) => (i === mega && p.startsWith('move') ? `${p} mega` : p)).join(', ')

  return (
    <div className="panel">
      <h3>Your move {step.forced && <small>(replacement: choose who comes in)</small>}</h3>
      {menu.map((m, i) => (
        <div key={i} className="row">
          <label>{m.actor ?? `Slot ${'AB'[i]}`}
            <select value={picks[i]} onChange={(e) => setPicks(picks.map((p, j) => (j === i ? e.target.value : p)))} disabled={m.options.length < 2}>
              {m.options.map((o) => <option key={o.choice} value={o.choice}>{o.label}{o.choice === suggested[i] ? '  ★ suggested' : ''}</option>)}
            </select>
          </label>
          {m.canMega && <label><input type="checkbox" checked={mega === i} onChange={(e) => setMega(e.target.checked ? i : -1)} /> Mega Evolve</label>}
        </div>
      ))}
      {sameSwitch && <p className="bad">Both slots can't switch to the same Pokémon.</p>}
      <div className="row">
        <button className="go" disabled={busy || sameSwitch} onClick={() => onPlay(choice)}>{busy ? 'Playing the turn…' : 'Play my move'}</button>
        <button disabled={busy} onClick={() => onPlay(step.choice)}>Play the suggestion</button>
      </div>
    </div>
  )
}

// Team preview: mark two of your six to lead and two to sit at the back; the other two stay home.
function Preview({ step, busy, onPlay }) {
  const fromPlan = (plan) => Object.fromEntries([...plan.lead.map((p) => [p, 'lead']), ...plan.back.map((p) => [p, 'back'])])
  const [roles, setRoles] = useState(fromPlan(step.plans[0]))
  const count = (r) => Object.values(roles).filter((x) => x === r).length
  const ok = count('lead') === 2 && count('back') === 2
  const order = (r) => step.ours.filter((m) => roles[m.position] === r).map((m) => m.position)
  const choice = `team ${[...order('lead'), ...order('back')].join('')}`
  const name = (p) => step.ours[p - 1].name
  const set = (position, role) => setRoles({ ...roles, [position]: roles[position] === role ? undefined : role })

  return (
    <>
      <div className="panel">
        <h3>Team preview <small>(choose your four: two to lead, two at the back)</small></h3>
        <p className="note">Their six: {step.theirs.join(' · ')}</p>
        <div className="preview-grid">
          {step.ours.map((m) => (
            <div key={m.position} className={`preview-mon ${roles[m.position] ?? 'home'}`}>
              <Sprite name={m.name} />
              <b>{m.name}</b>
              <small>{[m.item, m.ability].filter(Boolean).join(' · ')}</small>
              <small>{m.moves.join(', ')}</small>
              <div className="row">
                <button className={roles[m.position] === 'lead' ? 'on' : ''} onClick={() => set(m.position, 'lead')}>Lead</button>
                <button className={roles[m.position] === 'back' ? 'on' : ''} onClick={() => set(m.position, 'back')}>Back</button>
              </div>
            </div>
          ))}
        </div>
        {!ok && <p className="bad">Pick exactly two leads and two at the back (now {count('lead')} and {count('back')}).</p>}
        <div className="row">
          <button className="go" disabled={busy || !ok} onClick={() => onPlay(choice)}>{busy ? 'Starting…' : 'Bring these four'}</button>
          <button disabled={busy} onClick={() => onPlay(step.plans[0].choice)}>Use the model's top plan</button>
          <button disabled={busy} onClick={() => onPlay('default')}>Use the rule-based pick</button>
        </div>
      </div>
      <div className="panel">
        <h3>Coach: preview plans <small>(the model scores all 90 plans, a set of four and a lead pair inside it, against their six)</small></h3>
        <p className="note">Click a plan to load it. In testing the learned preview is still a few points weaker than the rule-based one.</p>
        {step.plans.map((p) => (
          <button key={p.choice} className="plan-pick" onClick={() => setRoles(fromPlan(p))}>
            <span>Lead <b>{p.lead.map(name).join(' + ')}</b> · back {p.back.map(name).join(' + ')}</span>
            <span className="barwrap"><i style={{ width: pct(p.prob / step.plans[0].prob) }} /></span>
            <span className="prob">{(p.prob * 100).toFixed(1)}%</span>
          </button>
        ))}
      </div>
    </>
  )
}

export default function Play() {
  const { options, error: feedError, choice, set } = useOptions()
  const [game, setGame] = useState(null)        // the server's last answer: {pending, result, error}
  const [steps, setSteps] = useState([])        // every move decision so far, for stepping back
  const [index, setIndex] = useState(0)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState(null)

  const take = (g, keep) => {
    if (g.error) throw new Error(g.error)
    setGame(g)
    const next = g.pending && g.pending.kind !== 'preview' ? [...keep, g.pending] : keep
    setSteps(next)
    if (g.pending && g.pending.kind !== 'preview') setIndex(next.length - 1)
  }
  const call = (url, keep) => {
    setBusy(true); setError(null)
    return fetch(url).then((r) => r.json()).then((g) => take(g, keep)).catch((e) => setError(String(e.message || e))).finally(() => setBusy(false))
  }
  const start = () => call(`/api/play/start?ours=${choice.ours}&opp=${choice.opp}&seed=${choice.seed}${choice.model ? `&model=${choice.model}` : ''}`, [])
  const play = (move) => call(`/api/play/move?choice=${encodeURIComponent(move)}`, steps.map((s, i) => (i === steps.length - 1 ? { ...s, played: move } : s)))

  if (feedError) return <div className="panel"><p>{feedError}</p></div>
  if (!options) return <p className="note">Setting up…</p>

  const step = game && game.pending
  const live = step && step.kind !== 'preview' && index === steps.length - 1
  const result = game && game.result
  return (
    <div className="wide-page">
      <Setup options={options} choice={choice} set={set} running={!!game} busy={busy} startLabel="Start game" onStart={start}
        summary={game ? `You play ${options.ours.find((t) => t.id === choice.ours)?.species.join(' / ') ?? 'your team'} · coach ${choice.model}${result ? ` · ${result.winner === 'you' ? 'you won' : result.winner === 'bot' ? 'you lost' : 'draw'} in ${result.turns} turns` : ''}` : ''} />
      {error && <p className="bad">{error}</p>}
      {step && step.kind === 'preview' && <Preview key={`preview-${steps.length}`} step={step} busy={busy} onPlay={play} />}
      {steps.length > 0 && (
        <Battle steps={steps} index={index} onIndex={setIndex}>
          {live && <YourMove key={`${step.board.turn}-${steps.length}`} step={step} busy={busy} onPlay={play} />}
          {!live && !result && <p className="note">Viewing an earlier decision. Jump to the last one to move.</p>}
          {result && <div className="panel"><h3>{result.winner === 'you' ? 'You won' : result.winner === 'bot' ? 'You lost' : 'Draw'} in {result.turns} turns</h3></div>}
        </Battle>
      )}
    </div>
  )
}
