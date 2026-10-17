import { useEffect, useState } from 'react'
import Sprite from './Sprite.jsx'

const pct = (x) => `${Math.round(x * 100)}%`
const NO_FEED = 'The match feed needs the venv Python: PYTHONPATH=src .venv/bin/python -m champions.ops_server'

function Side({ title, mons, hidden }) {
  return (
    <div className="side">
      <h4>{title}</h4>
      <div className="mons">
        {mons.map((m, i) => (
          <div key={i} className={`mon ${m.fainted ? 'down' : ''} ${m.active != null ? 'active' : ''} ${hidden && !m.seen ? 'unseen' : ''}`} title={[m.item, m.ability, ...m.moves].filter(Boolean).join(' · ')}>
            <Sprite name={m.name} />
            {hidden && !m.seen && <small>{m.name}: not seen yet</small>}
            {!(hidden && !m.seen) && <div className="hp"><i style={{ width: pct(m.hp), background: m.hp > 0.5 ? 'var(--o-good)' : m.hp > 0.2 ? 'var(--o-warn)' : 'var(--o-bad)' }} /></div>}
            {!(hidden && !m.seen) && <small>{m.fainted ? 'KO' : pct(m.hp)}{m.status ? ` ${m.status}` : ''}{m.active != null ? ` · slot ${'AB'[m.active]}` : ''}</small>}
            {Object.keys(m.boosts).length > 0 && <small className="boosts">{Object.entries(m.boosts).map(([k, v]) => `${k}${v > 0 ? '+' : ''}${v}`).join(' ')}</small>}
          </div>
        ))}
      </div>
    </div>
  )
}

function Field({ board }) {
  const chips = [board.trickRoom && 'Trick Room', board.weather, board.terrain && `${board.terrain} terrain`,
    ...board.ourConditions.map((c) => `ours: ${c}`), ...board.foeConditions.map((c) => `theirs: ${c}`)].filter(Boolean)
  return <div className="chips">{chips.length ? chips.map((c) => <span key={c} className={`chip ${c === 'Trick Room' ? 'tr' : ''}`}>{c}</span>) : <span className="note">no field effects</span>}</div>
}

function Option({ option, chosen, max }) {
  const [open, setOpen] = useState(false)
  return (
    <div className={`option ${chosen ? 'chosen' : ''}`}>
      <button onClick={() => option.damage && setOpen(!open)} aria-expanded={open}>
        <span className="label">{chosen ? '▶ ' : ''}{option.label}</span>
        <span className="barwrap"><i style={{ width: pct(option.prob / max) }} /></span>
        <span className="prob">{(option.prob * 100).toFixed(1)}%</span>
      </button>
      {open && option.damage && (
        <table className="dmg">
          <thead><tr><th>vs</th><th>type</th><th>takes</th><th>of its HP now</th><th>KO</th></tr></thead>
          <tbody>
            {option.damage.map((d) => (
              <tr key={d.foe}>
                <td>{d.foe}</td><td>×{d.effectiveness}</td><td>{pct(d.pctOfFullHp)}</td><td>{pct(d.pctOfCurrentHp)}</td>
                <td>{d.sureKo ? 'sure' : d.maybeKo ? 'possible' : '–'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  )
}

function Thought({ t }) {
  const max = Math.max(...t.options.map((o) => o.prob), 0.01)
  const damaging = t.options.filter((o) => o.damage).length
  return (
    <div className="thought">
      <div className="thought-head">
        <strong>{t.kind === 'replacement' ? 'Choosing a replacement' : `${t.actor} (slot ${'AB'[t.slot]})`}</strong>
        <span className="win" title="The value head's estimate of our chance to win from here">win chance {pct(t.winChance)}<i style={{ width: pct(t.winChance) }} /></span>
      </div>
      <p className="note">Foes on the field: {t.foes.filter(Boolean).join(' and ') || 'none'}. Scores are the policy's probabilities over legal actions{damaging ? '; click a move for the damage numbers it was shown' : ''}.</p>
      {t.options.map((o) => <Option key={o.label} option={o} chosen={o.label === t.chosen} max={max} />)}
      {t.target && (
        <p className="target">Target: {t.target.options.map((o) => `${o.foe} ${pct(o.prob)}`).join(' · ')} → <strong>{t.target.chosen ?? 'n/a'}</strong></p>
      )}
      {t.threat.length > 0 && (
        <details>
          <summary>Threat it sees against us</summary>
          <ul className="threat">
            {t.threat.map((x, i) => (
              <li key={i}>{x.foe} → {x.target}: strongest revealed move ≈ {pct(x.revealedMove)} of current HP, an unrevealed same-type attack ≈ {pct(x.guess)}</li>
            ))}
          </ul>
        </details>
      )}
    </div>
  )
}

export default function Match() {
  const [options, setOptions] = useState(null)
  const [ours, setOurs] = useState('')
  const [opp, setOpp] = useState('')
  const [seed, setSeed] = useState(1)
  const [sample, setSample] = useState(false)
  const [match, setMatch] = useState(null)
  const [step, setStep] = useState(0)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState(null)

  useEffect(() => {
    fetch('/api/match/options').then((r) => r.json()).then((o) => {
      if (o.error) throw new Error(o.error)
      setOptions(o); setOurs(o.ours[0]?.id ?? ''); setOpp(o.opponents[0]?.id ?? '')
    }).catch(() => setError(NO_FEED))
  }, [])

  useEffect(() => {
    if (!match) return undefined
    const onKey = (e) => {
      if (e.key === 'ArrowRight') setStep((s) => Math.min(s + 1, match.steps.length))
      if (e.key === 'ArrowLeft') setStep((s) => Math.max(s - 1, 0))
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [match])

  const simulate = (nextSeed = seed) => {
    setBusy(true); setError(null)
    fetch(`/api/match?ours=${ours}&opp=${opp}&seed=${nextSeed}&temp=${sample ? 1 : 0}`)
      .then((r) => r.json())
      .then((m) => { if (m.error) throw new Error(m.error); setMatch(m); setStep(0) })
      .catch((e) => setError(String(e.message || e)))
      .finally(() => setBusy(false))
  }
  const reroll = () => { const n = Math.floor(Math.random() * 100000); setSeed(n); simulate(n) }

  if (error && !options) return <div className="panel"><h3>Live surgery</h3><p>{error}</p></div>
  if (!options) return <p className="note">Preparing the theatre…</p>

  const atEnd = match && step === match.steps.length
  const current = match && (atEnd ? null : match.steps[step])
  const turns = match ? [...new Set(match.steps.map((s) => s.turn))] : []

  return (
    <div>
      <div className="panel controls">
        <h3>Live surgery <small>(one simulated match, step by step)</small></h3>
        <label>Our team (the learned player pilots it)
          <select value={ours} onChange={(e) => setOurs(e.target.value)}>
            {options.ours.map((t) => <option key={t.id} value={t.id}>{t.species.join(' / ')} ({t.label})</option>)}
          </select>
        </label>
        <label>Opponent (rule-based player)
          <select value={opp} onChange={(e) => setOpp(e.target.value)}>
            {options.opponents.map((t) => <option key={t.id} value={t.id}>{t.species.join(' / ')} ({t.label})</option>)}
          </select>
        </label>
        <div className="row">
          <label>Seed <input type="number" value={seed} onChange={(e) => setSeed(Number(e.target.value))} /></label>
          <label><input type="checkbox" checked={sample} onChange={(e) => setSample(e.target.checked)} /> sample from its scores instead of always playing the top choice</label>
        </div>
        <div className="row">
          <button className="go" disabled={busy || !ours || !opp} onClick={() => simulate()}>{busy ? 'Operating…' : 'Simulate match'}</button>
          <button disabled={busy || !ours || !opp} onClick={reroll}>New random game</button>
        </div>
        {error && <p className="bad">{error}</p>}
      </div>

      {match && (
        <>
          <div className="panel stepper">
            <div className="row">
              <button onClick={() => setStep(0)} disabled={step === 0}>⏮</button>
              <button onClick={() => setStep(step - 1)} disabled={step === 0}>◀</button>
              <span>Step {Math.min(step + 1, match.steps.length)} of {match.steps.length}{atEnd ? ' · result' : current ? ` · turn ${current.turn}` : ''}</span>
              <button onClick={() => setStep(step + 1)} disabled={atEnd}>▶</button>
              <button onClick={() => setStep(match.steps.length)} disabled={atEnd}>⏭</button>
            </div>
            <input type="range" min="0" max={match.steps.length} value={step} onChange={(e) => setStep(Number(e.target.value))} />
            <div className="chips">
              {turns.map((t) => (
                <button key={t} className={`chip clickable ${current && current.turn === t ? 'tr' : ''}`} onClick={() => setStep(match.steps.findIndex((s) => s.turn === t))}>turn {t}</button>
              ))}
              <button className={`chip clickable ${atEnd ? 'tr' : ''}`} onClick={() => setStep(match.steps.length)}>result</button>
            </div>
            <p className="note">Model {match.model} · seed {match.seed} · {match.temperature ? 'sampling' : 'top choice'} · arrow keys also step. We pilot {match.ours.join(' / ')}.</p>
          </div>

          {current ? (
            <>
              <div className="panel">
                <Field board={current.board} />
                <div className="board">
                  <Side title="Our four" mons={current.board.ours} />
                  <Side title="Their six (four will come)" mons={current.board.foe} hidden />
                </div>
              </div>
              {current.events.length > 0 && (
                <div className="panel">
                  <h3>What just happened</h3>
                  <ul className="events">{current.events.map((e, i) => <li key={i} className={e.tone}>{e.text}</li>)}</ul>
                </div>
              )}
              <div className="panel">
                <h3>What the model is thinking <small>(chose: {current.choice})</small></h3>
                {current.thoughts.map((t, i) => <Thought key={i} t={t} />)}
                {!current.thoughts.length && <p className="note">Nothing to decide (forced moves).</p>}
              </div>
            </>
          ) : (
            <div className="panel">
              <h3>Result: {match.winner === 'us' ? 'we won' : match.winner === 'foe' ? 'we lost' : 'draw'} in {match.turns} turns</h3>
              <ul className="events">{match.finalEvents.map((e, i) => <li key={i} className={e.tone}>{e.text}</li>)}</ul>
            </div>
          )}
        </>
      )}
    </div>
  )
}
