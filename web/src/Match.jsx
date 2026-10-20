// Watch a match: the model pilots one team against the rule-based bot; step through every decision.
import { useEffect, useState } from 'react'
import Battle from './Battle.jsx'
import { Setup, useOptions } from './GameSetup.jsx'

export default function Match() {
  const { options, error: feedError, choice, set } = useOptions()
  const [sample, setSample] = useState(false)
  const [search, setSearch] = useState(true)
  const [match, setMatch] = useState(null)
  const [index, setIndex] = useState(0)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState(null)

  useEffect(() => {
    if (!match) return undefined
    const onKey = (e) => {
      if (e.key === 'ArrowRight') setIndex((i) => Math.min(i + 1, match.steps.length - 1))
      if (e.key === 'ArrowLeft') setIndex((i) => Math.max(i - 1, 0))
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [match])

  const simulate = (seed = choice.seed) => {
    setBusy(true); setError(null)
    fetch(`/api/match?ours=${choice.ours}&opp=${choice.opp}&seed=${seed}&temp=${sample ? 1 : 0}&search=${search ? 8 : 0}${choice.model ? `&model=${choice.model}` : ''}`)
      .then((r) => r.json())
      .then((m) => { if (m.error) throw new Error(m.error); setMatch(m); setIndex(0) })
      .catch((e) => setError(String(e.message || e)))
      .finally(() => setBusy(false))
  }

  if (feedError) return <div className="panel"><p>{feedError}</p></div>
  if (!options) return <p className="note">Setting up…</p>
  const result = match && `${match.winner === 'us' ? 'We won' : match.winner === 'foe' ? 'We lost' : 'Draw'} in ${match.turns} turns`

  return (
    <div className="wide-page">
      <Setup options={options} choice={choice} set={set} running={!!match} busy={busy} startLabel="Simulate match" onStart={() => simulate()}
        summary={match ? `${match.model} · seed ${match.seed} · ${match.ours.join(' / ')} · ${result}` : ''}>
        <label><input type="checkbox" checked={sample} onChange={(e) => setSample(e.target.checked)} /> sample from its scores</label>
        <label><input type="checkbox" checked={search} onChange={(e) => setSearch(e.target.checked)} /> one-turn search</label>
      </Setup>
      {error && <p className="bad">{error}</p>}
      {match && match.steps.length > 0 && <Battle steps={match.steps} index={index} onIndex={setIndex} />}
    </div>
  )
}
