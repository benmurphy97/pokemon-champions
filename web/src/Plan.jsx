import { useEffect, useState } from 'react'

const pct = (x) => (x == null ? '–' : `${(x * 100).toFixed(1)}%`)
const ago = (t) => {
  const s = Date.now() / 1000 - t
  return s < 3600 ? `${Math.round(s / 60)} min ago` : s < 172800 ? `${Math.round(s / 3600)} h ago` : `${Math.round(s / 86400)} d ago`
}
const ICON = { done: '✔', doing: '◐', todo: '○' }

function Gates({ plan }) {
  return (
    <div className="panel">
      <h3>Gates <small>(what "playable" and "good" mean)</small></h3>
      {plan.gates.map((g) => {
        const rows = plan.results.filter((r) => r.gate === g.id)
        const best = rows.reduce((a, r) => (!a || r.value > a.value ? r : a), null)
        const passed = best && best.value >= g.threshold
        return (
          <div className="gate" key={g.id}>
            <div className="gate-head">
              <strong>{g.name}</strong>
              <span className={`badge ${passed ? 'ok' : 'no'}`}>{best ? (passed ? `passed · ${best.model}` : `best so far ${pct(best.value)}`) : 'not measured yet'}</span>
            </div>
            <p className="note">{g.text}</p>
            <div className="meter"><i style={{ width: `${(best ? best.value : 0) * 100}%` }} /><b style={{ left: `${g.threshold * 100}%` }} title={`threshold ${pct(g.threshold)}`} /></div>
            {rows.length > 0 && (
              <table>
                <thead><tr><th>Model</th><th>Result</th><th>95% range</th><th>Games</th><th>When</th></tr></thead>
                <tbody>
                  {[...rows].reverse().slice(0, 6).map((r, i) => (
                    <tr key={i}>
                      <td>{r.model}</td>
                      <td className={r.value >= g.threshold ? 'okc' : ''}>{pct(r.value)}{r.baseline != null && <span className="note"> (rule-based {pct(r.baseline)})</span>}</td>
                      <td>{pct(r.interval[0])}–{pct(r.interval[1])}</td><td>{r.games}</td><td>{ago(r.time)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>
        )
      })}
      <p className="note">Record a model: <code>PYTHONPATH=src .venv/bin/python -m champions.gates --model models/NAME.pt</code></p>
    </div>
  )
}

function Checklist({ title, items, label }) {
  return (
    <div className="panel">
      <h3>{title}</h3>
      <ul className="checklist">
        {items.map((it, i) => (
          <li key={i} className={it.status}>
            <span className="icon">{ICON[it.status]}</span>
            <div><strong>{label(it)}</strong>{it.note && <p className="note">{it.note}</p>}</div>
          </li>
        ))}
      </ul>
    </div>
  )
}

function Models({ models }) {
  return (
    <div className="panel">
      <h3>Model registry <small>(everything in models/)</small></h3>
      <div className="scroll">
        <table>
          <thead><tr><th>Model</th><th>Size</th><th>Joint choice</th><th>Imitation top-1</th><th>Latest held-out check</th><th>Saved</th></tr></thead>
          <tbody>
            {models.map((m) => (
              <tr key={m.name}>
                <td><strong>{m.name}</strong><br /><span className="note">{(m.parameters / 1e6).toFixed(2)}M params · {m.config.layers}×{m.config.dim}{m.encodingVersion ? ` · encoding v${m.encodingVersion}` : ' · older checkpoint'}</span></td>
                <td>{m.sizeMb} MB</td>
                <td>{m.jointChoice ? <span className="badge ok">trained</span> : <span className="badge no">not yet</span>}</td>
                <td>{pct(m.imitationTop1)}</td>
                <td>{m.heldOut ? `${pct(m.heldOut.policy)} vs ${pct(m.heldOut.baseline)}` : '–'}{m.iteration ? <span className="note"> · iteration {m.iteration}</span> : null}</td>
                <td>{ago(m.modified)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="note">"Joint choice" means the second Pokémon's choice was learned knowing the first's. Older models still load and play, with that input inert.</p>
    </div>
  )
}

export default function Plan() {
  const [plan, setPlan] = useState(null)
  const [error, setError] = useState(null)
  useEffect(() => {
    let live = true
    const load = () => fetch('/api/plan').then((r) => r.json()).then((p) => { if (live) { if (p.error) setError(p.error); else { setPlan(p); setError(null) } } }).catch(() => live && setError('feed offline'))
    load()
    const id = setInterval(load, 10000)
    return () => { live = false; clearInterval(id) }
  }, [])
  if (error && !plan) return <div className="panel"><h3>Plan unavailable</h3><p>{error}. Start the feed with the venv Python: <code>PYTHONPATH=src .venv/bin/python -m champions.ops_server</code></p></div>
  if (!plan) return <p className="note">Loading the plan…</p>
  return (
    <div>
      <Gates plan={plan} />
      <div className="grid2">
        <Checklist title="Frozen design" items={plan.frozen} label={(it) => it.name} />
        <Checklist title="Order of work" items={plan.steps} label={(it) => `${it.n}. ${it.name}`} />
      </div>
      <Models models={plan.models} />
      <div className="panel">
        <h3>Process rules</h3>
        <ul className="rules">{plan.rules.map((r) => <li key={r}>{r}</li>)}</ul>
        <p className="note">Full plan: docs/plan.md · how to extend the model: docs/extending.md · status updated {plan.updated}</p>
      </div>
    </div>
  )
}
