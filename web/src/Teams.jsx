import { useMemo, useState } from 'react'
import Sprite from './Sprite.jsx'
import { winRate } from './rating.js'

const PAGE = 30

export function Team({ team, rank }) {
  const [open, setOpen] = useState(false)
  return (
    <article className="card">
      <button className="card-head" onClick={() => setOpen(!open)} aria-expanded={open}>
        <div className="sprites">{team.mons.map((m) => <Sprite key={m.name} name={m.name} />)}</div>
        <div className="meta">
          {rank && <strong>#{rank} · </strong>}
          {team.rating && <strong title={`${team.rating.wins} wins in ${team.rating.games} simulated games`}>{winRate(team.rating)} simulated · </strong>}
          <strong>{team.placement ?? '–'}</strong> · {team.tournament}
          <span className="muted"> · {team.date}</span>
          <div className="tags">{team.tags.map((t) => <span key={t} className="tag">{t}</span>)}</div>
        </div>
      </button>
      {open && (
        <div className="sets">
          {team.mons.map((m) => (
            <div key={m.name} className="set">
              <h3>{m.name} <span className="muted">@ {m.item}</span></h3>
              <p className="muted">{m.ability} · {m.nature}</p>
              <p className="muted">{Object.entries(m.points).map(([k, v]) => `${v} ${k}`).join(' / ')}</p>
              <ul>{m.moves.map((mv) => <li key={mv}>{mv}</li>)}</ul>
            </div>
          ))}
        </div>
      )}
    </article>
  )
}

export default function Teams({ teams }) {
  const [search, setSearch] = useState('')
  const [tag, setTag] = useState('')
  const [goodOnly, setGoodOnly] = useState(false)
  const [shown, setShown] = useState(PAGE)

  const tags = useMemo(() => [...new Set(teams.flatMap((t) => t.tags))].sort(), [teams])
  const filtered = useMemo(() => {
    const q = search.trim().toLowerCase()
    return teams.filter((t) =>
      (!goodOnly || t.good) &&
      (!tag || t.tags.includes(tag)) &&
      (!q || t.mons.some((m) => m.name.toLowerCase().includes(q))))
  }, [teams, search, tag, goodOnly])

  return (
    <section>
      <div className="filters">
        <input placeholder="Search a Pokémon…" value={search} onChange={(e) => { setSearch(e.target.value); setShown(PAGE) }} />
        <select value={tag} onChange={(e) => { setTag(e.target.value); setShown(PAGE) }}>
          <option value="">Any strategy</option>
          {tags.map((t) => <option key={t}>{t}</option>)}
        </select>
        <label><input type="checkbox" checked={goodOnly} onChange={(e) => { setGoodOnly(e.target.checked); setShown(PAGE) }} /> Top finishes only</label>
      </div>
      <p className="muted">{filtered.length} teams</p>
      {filtered.slice(0, shown).map((t) => <Team key={t.id} team={t} />)}
      {shown < filtered.length && <button className="more" onClick={() => setShown(shown + PAGE)}>Show more</button>}
    </section>
  )
}
