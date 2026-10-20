import { useEffect, useState } from 'react'
import Best from './Best.jsx'
import Model from './Model.jsx'
import Match from './Match.jsx'
import Ops from './Ops.jsx'
import Play from './Play.jsx'
import Strategies from './Strategies.jsx'
import Teams from './Teams.jsx'
import Usage from './Usage.jsx'

const load = (name) => fetch(`./data/${name}.json`).then((r) => r.json())

export default function App() {
  const [data, setData] = useState(null)
  const [error, setError] = useState(null)
  const [tab, setTab] = useState('ops')

  useEffect(() => {
    Promise.all([load('meta'), load('teams'), load('usage'), load('strategies'), load('model')])
      .then(([meta, teams, usage, strategies, model]) => setData({ meta, teams, usage, strategies, model }))
      .catch((e) => setError(String(e)))
  }, [])

  if (error) return <p className="notice">Could not load data: {error}. Run the exporter first (see web/README.md).</p>
  if (!data) return <p className="notice">Loading…</p>

  return (
    <main className={tab === 'watch' || tab === 'play' ? 'wide' : ''}>
      <header>
        <h1>Pokémon Champions teams</h1>
        <p className="muted">Regulation {data.meta.regulation} · {data.meta.teams} tournament teams</p>
        <nav>
          <button className={tab === 'ops' ? 'on' : ''} onClick={() => setTab('ops')}>Operating room</button>
          <button className={tab === 'watch' ? 'on' : ''} onClick={() => setTab('watch')}>Watch a match</button>
          <button className={tab === 'play' ? 'on' : ''} onClick={() => setTab('play')}>Play with the coach</button>
          <button className={tab === 'best' ? 'on' : ''} onClick={() => setTab('best')}>Best teams</button>
          <button className={tab === 'teams' ? 'on' : ''} onClick={() => setTab('teams')}>All teams</button>
          <button className={tab === 'strategies' ? 'on' : ''} onClick={() => setTab('strategies')}>Strategies</button>
          <button className={tab === 'model' ? 'on' : ''} onClick={() => setTab('model')}>Model</button>
          <button className={tab === 'usage' ? 'on' : ''} onClick={() => setTab('usage')}>Pokémon</button>
        </nav>
      </header>
      {tab === 'ops' && <Ops />}
      {tab === 'watch' && <section className="ops"><Match /></section>}
      {tab === 'play' && <section className="ops"><Play /></section>}
      {tab === 'best' && <Best teams={data.teams} />}
      {tab === 'teams' && <Teams teams={data.teams} />}
      {tab === 'strategies' && <Strategies strategies={data.strategies} />}
      {tab === 'model' && <Model model={data.model} />}
      {tab === 'usage' && <Usage usage={data.usage} />}
    </main>
  )
}
