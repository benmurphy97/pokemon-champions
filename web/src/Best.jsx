import { useMemo, useState } from 'react'
import { Team } from './Teams.jsx'
import { lowerBound } from './rating.js'

export default function Best({ teams }) {
  const [tag, setTag] = useState('')
  const rated = useMemo(() => teams.filter((t) => t.rating), [teams])
  const tags = useMemo(() => [...new Set(rated.flatMap((t) => t.tags))].sort(), [rated])
  const ranked = useMemo(
    () => rated
      .filter((t) => !tag || t.tags.includes(tag))
      .sort((a, b) => lowerBound(b.rating.wins, b.rating.games) - lowerBound(a.rating.wins, a.rating.games))
      .slice(0, 25),
    [rated, tag])

  return (
    <section>
      <p className="muted">
        Teams ranked by simulated win rate against 60 real ladder teams, with the same player piloting both sides
        (a rule-based player, for now). Ranked by the low end of the 95% range, so a team needs plenty of games to rank high.
      </p>
      <div className="filters">
        <select value={tag} onChange={(e) => setTag(e.target.value)}>
          <option value="">Any strategy</option>
          {tags.map((t) => <option key={t}>{t}</option>)}
        </select>
      </div>
      {ranked.map((t, i) => <Team key={t.id} team={t} rank={i + 1} />)}
      {!ranked.length && <p className="notice">No rated teams yet. Run champions.rate_teams, then the exporter.</p>}
    </section>
  )
}
