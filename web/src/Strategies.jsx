const pct = (x) => `${Math.round(x * 100)}%`
const list = (pairs) => pairs.map(([name, share]) => `${name} ${pct(share)}`).join(', ')

export default function Strategies({ strategies }) {
  return (
    <section>
      <p className="muted">How teams in each style play and how they fare in simulation. Trick Room is the style our player is trained on.</p>
      {strategies.map((s) => (
        <article key={s.tag} className="card strategy">
          <h3>{s.tag}{s.tag === 'Trick Room' && <span className="tag"> trained</span>}</h3>
          <p className="muted">{s.teams} teams · {s.top8} top finishes · {pct(s.simulatedWinRate)} simulated win rate</p>
          {s.setters && <p><strong>Setters:</strong> {list(s.setters)}</p>}
          <p><strong>Common Pokémon:</strong> {list(s.pokemon)}</p>
        </article>
      ))}
    </section>
  )
}
