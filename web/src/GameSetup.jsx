import { useEffect, useState } from 'react'

export const NO_FEED = 'This page needs the Python feed: PYTHONPATH=src .venv/bin/python -m champions.site.ops_server'

// Teams and models the server offers, with sensible defaults chosen.
export function useOptions() {
  const [options, setOptions] = useState(null)
  const [error, setError] = useState(null)
  const [choice, setChoice] = useState({ ours: '', opp: '', model: '', seed: 1 })
  useEffect(() => {
    fetch('/api/match/options').then((r) => r.json()).then((o) => {
      if (o.error) throw new Error(o.error)
      setOptions(o)
      setChoice((c) => ({ ...c, ours: o.ours[0]?.id ?? '', opp: o.opponents[0]?.id ?? '', model: ['rl_run9_best.pt', 'rl_policy.pt'].find((m) => o.models.includes(m)) ?? o.models[0] ?? '' }))
    }).catch(() => setError(NO_FEED))
  }, [])
  return { options, error, choice, set: (patch) => setChoice((c) => ({ ...c, ...patch })) }
}

// The new-game form. While a game is running it folds into one line, so the battle gets the screen.
export function Setup({ options, choice, set, running, summary, busy, onStart, startLabel, children }) {
  const [open, setOpen] = useState(false)
  if (running && !open) {
    return (
      <div className="slim row">
        <span className="note">{summary}</span>
        <button onClick={() => setOpen(true)}>New game…</button>
      </div>
    )
  }
  return (
    <div className="panel controls">
      <div className="row">
        <label>Your team
          <select value={choice.ours} onChange={(e) => set({ ours: e.target.value })}>
            {options.ours.map((t) => <option key={t.id} value={t.id}>{t.species.join(' / ')} ({t.label})</option>)}
          </select>
        </label>
        <label>Opponent team
          <select value={choice.opp} onChange={(e) => set({ opp: e.target.value })}>
            {options.opponents.map((t) => <option key={t.id} value={t.id}>{t.species.join(' / ')} ({t.label})</option>)}
          </select>
        </label>
      </div>
      <div className="row">
        <label>Model
          <select value={choice.model} onChange={(e) => set({ model: e.target.value })}>{options.models.map((m) => <option key={m}>{m}</option>)}</select>
        </label>
        <label>Seed <input type="number" value={choice.seed} onChange={(e) => set({ seed: Number(e.target.value) })} /></label>
        {children}
      </div>
      <div className="row">
        <button className="go" disabled={busy || !choice.ours || !choice.opp} onClick={() => { setOpen(false); onStart() }}>{busy ? 'Starting…' : startLabel}</button>
        {running && <button onClick={() => setOpen(false)}>Cancel</button>}
      </div>
    </div>
  )
}
