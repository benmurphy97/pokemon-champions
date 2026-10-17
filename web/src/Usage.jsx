import { useState } from 'react'
import Sprite from './Sprite.jsx'

const pct = (x) => `${Math.round(x * 100)}%`
const list = (pairs) => pairs.map(([name, share]) => `${name} ${pct(share)}`).join(', ')

export default function Usage({ usage }) {
  const [search, setSearch] = useState('')
  const rows = usage.pokemon.filter((p) => p.name.toLowerCase().includes(search.trim().toLowerCase()))
  return (
    <section>
      <div className="filters">
        <input placeholder="Search a Pokémon…" value={search} onChange={(e) => setSearch(e.target.value)} />
      </div>
      <p className="muted">Ladder usage, {usage.month} ({usage.battles.toLocaleString()} battles)</p>
      {rows.map((p) => (
        <article key={p.name} className="card usage">
          <Sprite name={p.name} />
          <div>
            <strong>{p.name}</strong> <span className="muted">{pct(p.usage)} of teams</span>
            <p className="muted">Moves: {list(p.moves)}</p>
            <p className="muted">Items: {list(p.items)} · Ability: {list(p.abilities)}</p>
            <p className="muted">Often with: {list(p.partners)}</p>
          </div>
        </article>
      ))}
    </section>
  )
}
