import { useEffect, useState } from 'react'
import { Field, Plans, SearchTable, Side } from './Match.jsx'
import Sprite from './Sprite.jsx'

const NO_FEED = 'Play mode needs the venv Python: PYTHONPATH=src .venv/bin/python -m champions.site.ops_server'
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

function Coaching({ step }) {
  const thought = step.thoughts.find((t) => t.foePrediction && t.foePrediction.length) ?? step.thoughts[0]
  const pick = step.analysis && step.analysis.rows.find((r) => r.choice === step.analysis.choice)
  return (
    <div className="panel">
      <h3>Coach</h3>
      <p className="decision">Suggested: <b>{pick ? pick.label : step.choice}</b>
        {thought && <span className="note"> · its estimate of your win chance: {pct(thought.winChance)}</span>}</p>
      {thought && thought.foePrediction && thought.foePrediction.length > 0 && (
        <>
          <h4>What it expects the opponent to do</h4>
          <ul className="threat">
            {thought.foePrediction.map((f) => <li key={f.foe}><b>{f.foe}</b>: {f.top.map((x) => `${x.label} ${pct(x.prob)}`).join(' · ')}</li>)}
          </ul>
        </>
      )}
      <SearchTable analysis={step.analysis} policyChoice={step.policyChoice} />
      <details><summary>The policy's own scores before search</summary><Plans plans={step.plans} /></details>
    </div>
  )
}

export default function Play() {
  const [options, setOptions] = useState(null)
  const [ours, setOurs] = useState('')
  const [opp, setOpp] = useState('')
  const [model, setModel] = useState('')
  const [seed, setSeed] = useState(1)
  const [game, setGame] = useState(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState(null)

  useEffect(() => {
    fetch('/api/match/options').then((r) => r.json()).then((o) => {
      if (o.error) throw new Error(o.error)
      setOptions(o); setOurs(o.ours[0]?.id ?? ''); setOpp(o.opponents[0]?.id ?? '')
      setModel(['rl_run9_best.pt', 'rl_policy.pt'].find((m) => o.models.includes(m)) ?? o.models[0] ?? '')
    }).catch(() => setError(NO_FEED))
    fetch('/api/play/state').then((r) => r.json()).then((g) => { if (g.pending || g.result) setGame(g) }).catch(() => {})
  }, [])

  const call = (url) => {
    setBusy(true); setError(null)
    fetch(url).then((r) => r.json())
      .then((g) => { if (g.error) throw new Error(g.error); setGame(g) })
      .catch((e) => setError(String(e.message || e)))
      .finally(() => setBusy(false))
  }
  const start = () => call(`/api/play/start?ours=${ours}&opp=${opp}&seed=${seed}${model ? `&model=${model}` : ''}`)
  const play = (choice) => call(`/api/play/move?choice=${encodeURIComponent(choice)}`)

  if (error && !options) return <div className="panel"><h3>Play</h3><p>{error}</p></div>
  if (!options) return <p className="note">Setting up the table…</p>

  const step = game && game.pending
  return (
    <div>
      <div className="panel controls">
        <h3>Play against the bot <small>(you pilot your team; the model coaches every decision)</small></h3>
        <label>Your team
          <select value={ours} onChange={(e) => setOurs(e.target.value)}>
            {options.ours.map((t) => <option key={t.id} value={t.id}>{t.species.join(' / ')} ({t.label})</option>)}
          </select>
        </label>
        <label>Opponent team (piloted by greedy-blind)
          <select value={opp} onChange={(e) => setOpp(e.target.value)}>
            {options.opponents.map((t) => <option key={t.id} value={t.id}>{t.species.join(' / ')} ({t.label})</option>)}
          </select>
        </label>
        <div className="row">
          <label>Coach model
            <select value={model} onChange={(e) => setModel(e.target.value)}>{options.models.map((m) => <option key={m}>{m}</option>)}</select>
          </label>
          <label>Seed <input type="number" value={seed} onChange={(e) => setSeed(Number(e.target.value))} /></label>
        </div>
        <button className="go" disabled={busy || !ours || !opp} onClick={start}>{busy && !game ? 'Dealing…' : game ? 'New game' : 'Start game'}</button>
        {error && <p className="bad">{error}</p>}
      </div>

      {game && game.history.length > 0 && (
        <div className="panel">
          <h3>So far</h3>
          {game.history.map((h, i) => (
            <div key={i} className="history">
              <ul className="events">{h.events.map((e, j) => <li key={j} className={e.tone}>{e.text}</li>)}</ul>
              {h.turn === 0 && <p className="note">Team preview</p>}
              {h.played && <p className="note">You played <b>{h.played}</b>{h.played === h.suggested ? ' (the suggestion)' : ` · coach suggested ${h.suggested}`}</p>}
            </div>
          ))}
        </div>
      )}

      {step && step.kind === 'preview' && <Preview key={`preview-${game.history.length}`} step={step} busy={busy} onPlay={play} />}

      {step && step.kind !== 'preview' && (
        <>
          <div className="panel">
            <p className="note">Turn {step.board.turn}</p>
            <Field board={step.board} />
            <div className="board">
              <Side title="Your four" mons={step.board.ours} />
              <div className="vs">VS</div>
              <Side title="Their six (four will come)" mons={step.board.foe} hidden />
            </div>
          </div>
          {step.events.length > 0 && (
            <div className="panel"><h3>What just happened</h3>
              <ul className="events">{step.events.map((e, i) => <li key={i} className={e.tone}>{e.text}</li>)}</ul>
            </div>
          )}
          <YourMove key={`${step.board.turn}-${game.history.length}`} step={step} busy={busy} onPlay={play} />
          <Coaching step={step} />
        </>
      )}

      {game && game.result && (
        <div className="panel">
          <h3>{game.result.winner === 'you' ? 'You won' : game.result.winner === 'bot' ? 'You lost' : 'Draw'} in {game.result.turns} turns</h3>
          <ul className="events">{game.result.finalEvents.map((e, i) => <li key={i} className={e.tone}>{e.text}</li>)}</ul>
        </div>
      )}
    </div>
  )
}
