import { useEffect, useState } from 'react'
import { Chart, rolling } from './charts.jsx'
import Match from './Match.jsx'
import Sprite from './Sprite.jsx'
import { winRate } from './rating.js'
import './ops.css'

const GAMES_PER_ITERATION = 256
const COLORS = { blind: '#4dd4ac', greedy: '#ffb347', random: '#7aa2ff', bad: '#ff6b6b', good: '#4dd4ac', warn: '#ffb347' }

const duration = (seconds) => {
  if (!isFinite(seconds)) return '–'
  const h = Math.floor(seconds / 3600), m = Math.round((seconds % 3600) / 60)
  return h ? `${h}h ${m}m` : `${m}m`
}
const ago = (seconds) => (seconds < 90 ? `${Math.round(seconds)}s ago` : `${Math.round(seconds / 60)} min ago`)
const mean = (xs) => xs.reduce((a, b) => a + b, 0) / Math.max(xs.length, 1)

function usePoll(url, ms) {
  const [data, setData] = useState(null)
  const [offline, setOffline] = useState(false)
  useEffect(() => {
    let live = true
    const tick = () => fetch(url).then((r) => r.json()).then((d) => { if (live) { setData(d); setOffline(false) } })
      .catch(() => live && setOffline(true))
    tick()
    const id = setInterval(tick, ms)
    return () => { live = false; clearInterval(id) }
  }, [url, ms])
  return { data, offline }
}

function Heartbeat({ running, stalled }) {
  const state = !running ? 'flat' : stalled ? 'slow' : 'beat'
  return (
    <div className={`ecg ${state}`} aria-hidden="true">
      <svg viewBox="0 0 1200 60" preserveAspectRatio="none">
        <path id="beat" className="trace" d="M0,30 L60,30 L70,28 L80,30 L100,30 L108,36 L116,4 L124,52 L132,30 L170,30 L184,22 L200,30 L300,30 L360,30 L370,28 L380,30 L400,30 L408,36 L416,4 L424,52 L432,30 L470,30 L484,22 L500,30 L600,30" />
        <path className="trace" transform="translate(600 0)" d="M0,30 L60,30 L70,28 L80,30 L100,30 L108,36 L116,4 L124,52 L132,30 L170,30 L184,22 L200,30 L300,30 L360,30 L370,28 L380,30 L400,30 L408,36 L416,4 L424,52 L432,30 L470,30 L484,22 L500,30 L600,30" />
      </svg>
    </div>
  )
}

function Vital({ label, value, sub, tone }) {
  return (
    <div className={`vital ${tone || ''}`}>
      <span className="vital-label">{label}</span>
      <strong>{value}</strong>
      {sub && <span className="vital-sub">{sub}</span>}
    </div>
  )
}

function alerts(run, training, now) {
  const out = []
  const rows = run.iterations
  const last = rows[rows.length - 1]
  if (training.running && last && now - run.updated > 20 * 60) out.push({ level: 'bad', text: `No new iteration for ${Math.round((now - run.updated) / 60)} min: training may be stuck.` })
  for (const e of run.evals.slice(-3)) {
    const gap = (e.policy - e.baseline) * 100
    const text = `Check at iteration ${e.iteration}: ${(e.policy * 100).toFixed(1)}% vs rule-based ${(e.baseline * 100).toFixed(1)}% (${gap >= 0 ? '+' : ''}${gap.toFixed(1)} pts, 95% range ${(e.interval[0] * 100).toFixed(0)}–${(e.interval[1] * 100).toFixed(0)})`
    const lowBeats = e.interval[0] > e.baseline, highBelow = e.interval[1] < e.baseline
    out.push({ level: lowBeats ? 'good' : highBelow ? 'bad' : 'warn', text: text + (lowBeats ? ': clearly ahead.' : highBelow ? ': clearly behind.' : ': within noise.') })
    if (e.invalid_choices > 5) out.push({ level: 'warn', text: `${e.invalid_choices} invalid choices at iteration ${e.iteration}.` })
  }
  if (last && last.kl > 0.2) out.push({ level: 'warn', text: `KL ${last.kl.toFixed(2)}: drifting far from the starting policy.` })
  if (last && last.entropy < 0.3) out.push({ level: 'warn', text: `Entropy ${last.entropy.toFixed(2)}: the policy is becoming very deterministic.` })
  if (!out.length) out.push({ level: 'good', text: 'All quiet. No checks yet.' })
  return out.reverse()
}

function Monitors({ run }) {
  const rows = run.iterations
  const series = (key) => rolling(rows.filter((r) => r.win_rate[key] != null).map((r) => [r.iteration, r.win_rate[key]]))
  const metric = (key) => rows.map((r) => [r.iteration, r[key]])
  const evals = run.evals
  return (
    <div className="grid2">
      <div className="panel">
        <h3>Win rate in training games <small>(5-iteration average)</small></h3>
        <Chart
          yMin={0} yMax={1} format={(v) => `${Math.round(v * 100)}%`}
          series={[
            { name: 'vs greedy-blind', color: COLORS.blind, points: series('greedy-blind') },
            { name: 'vs greedy', color: COLORS.greedy, points: series('greedy') },
            { name: 'vs random', color: COLORS.random, points: series('random') },
          ]}
          refs={[{ y: 0.5, label: '50%', color: '#6b7a86' }]}
        />
      </div>
      <div className="panel">
        <h3>Held-out check <small>(1,200 games, shaded = 95% range)</small></h3>
        <Chart
          yMin={0.3} yMax={0.8} format={(v) => `${Math.round(v * 100)}%`}
          series={[
            { name: 'learned player', color: COLORS.blind, points: evals.map((e) => [e.iteration, e.policy]), dots: true },
            { name: 'rule-based player', color: COLORS.greedy, dashed: true, points: evals.map((e) => [e.iteration, e.baseline]) },
          ]}
          band={{ color: COLORS.blind, points: evals.map((e) => [e.iteration, e.interval[0], e.interval[1]]) }}
        />
      </div>
      <div className="panel">
        <h3>Drift from start <small>(KL)</small></h3>
        <Chart height={110} series={[{ name: 'KL', color: COLORS.warn, points: rolling(metric('kl'), 3) }]} />
      </div>
      <div className="panel">
        <h3>Randomness <small>(entropy)</small></h3>
        <Chart height={110} series={[{ name: 'entropy', color: COLORS.random, points: rolling(metric('entropy'), 3) }]} />
      </div>
    </div>
  )
}

function Compare({ runs }) {
  const palette = ['#7aa2ff', '#c792ea', '#ffb347', '#ff7ab6', '#4dd4ac', '#e6e6e6']
  const withEvals = runs.filter((r) => r.evals.length)
  const baseline = mean(withEvals.flatMap((r) => r.evals.map((e) => e.baseline)))
  return (
    <div className="panel">
      <h3>Case notes: every run, held-out checks</h3>
      <Chart
        yMin={0.3} yMax={0.8} format={(v) => `${Math.round(v * 100)}%`}
        series={withEvals.map((r, i) => ({ name: r.name === 'current' ? 'current run' : r.name, color: palette[i % palette.length], dots: true, points: r.evals.map((e) => [e.iteration, e.policy]) }))}
        refs={[{ y: baseline, label: 'rule-based', color: '#6b7a86' }]}
      />
      <p className="note">Runs differ in setup (see docs/learned-agent.md); the "current" run may use different test teams from older ones, so compare each run with its own baseline.</p>
    </div>
  )
}

function Roster({ roster }) {
  if (!roster.length) return null
  return (
    <div className="panel">
      <h3>Patients <small>(the known-good Trick Room teams in training)</small></h3>
      <div className="roster">
        {roster.map((t) => (
          <div className="chart-card" key={t.id}>
            <div className="sprites">{t.species.map((s) => <Sprite key={s} name={s} />)}</div>
            <div className="card-meta">
              <strong>{t.placement}</strong> · {t.tournament}
              {t.rating && <span> · simulated {winRate(t.rating)} ({t.rating.games} games)</span>}
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

function Theatre({ jobs, load, cores }) {
  return (
    <div className="panel">
      <h3>Theatre <small>(running jobs · load {load[0].toFixed(1)} of {cores} cores)</small></h3>
      {jobs.length === 0 && <p className="note">Nothing is running.</p>}
      {jobs.map((j) => (
        <div className="job" key={j.pid}>
          <span className="dot" />
          <div><strong>{j.label}</strong> <span className="note">pid {j.pid} · up {j.elapsed} · {j.memoryMb} MB</span><br /><code>{j.command.slice(0, 90)}</code></div>
          <div className="cpu"><i style={{ width: `${Math.min(100, j.cpu / (cores * 10))}%` }} /></div>
        </div>
      ))}
    </div>
  )
}

export default function Ops() {
  const { data, offline } = usePoll('/api/ops', 4000)
  const [picked, setPicked] = useState(null)
  const [view, setView] = useState('vitals')

  if (offline && !data) {
    return <section className="ops"><div className="panel"><h3>Feed offline</h3><p>Start the monitor feed from the repo root:</p><code>PYTHONPATH=src python3 -m champions.ops_server</code></div></section>
  }
  if (!data) return <section className="ops"><p className="note">Connecting to the ward…</p></section>

  const run = data.runs.find((r) => r.name === picked) ?? data.runs[data.runs.length - 1]
  const rows = run.iterations
  const live = run.live && data.training.running
  const total = live ? data.training.totalIterations : rows.length
  const last = rows[rows.length - 1]
  const recent = rows.slice(-5).map((r) => r.rollout_seconds + r.update_seconds)
  const perIteration = mean(recent) * 1.05          // checks every 10 iterations add a few percent
  const remaining = Math.max(total - rows.length, 0)
  const stalled = live && data.now - run.updated > 20 * 60
  const progress = total ? rows.length / total : 0

  return (
    <section className="ops">
      <div className="panel hero">
        <div>
          <span className={`status ${live ? (stalled ? 'warn' : 'live') : 'idle'}`}>
            <i /> {live ? (stalled ? 'STALLED?' : 'TRAINING LIVE') : 'NO ACTIVE SURGERY'}
          </span>
          <h2>Operating room</h2>
          <p className="note">{live ? data.training.command : `Showing finished run ${run.name}.`}</p>
        </div>
        <Heartbeat running={live} stalled={stalled} />
      </div>

      <div className="tabs views">
        <button className={view === 'vitals' ? 'on' : ''} onClick={() => setView('vitals')}>Vitals & monitors</button>
        <button className={view === 'surgery' ? 'on' : ''} onClick={() => setView('surgery')}>Live surgery (match viewer)</button>
      </div>
      {view === 'surgery' ? <Match /> : <>
      <div className="tabs">
        {data.runs.map((r) => (
          <button key={r.name} className={r.name === run.name ? 'on' : ''} onClick={() => setPicked(r.name)}>
            {r.name === 'current' ? (data.training.running ? '● current' : 'latest') : r.name}
          </button>
        ))}
      </div>

      <div className="vitals">
        <Vital label="Iteration" value={`${rows.length}/${total}`} sub={<span className="bar"><i style={{ width: `${progress * 100}%` }} /></span>} />
        <Vital label="Games played" value={(rows.length * GAMES_PER_ITERATION).toLocaleString()} sub={`${last ? last.decisions.toLocaleString() : 0} decisions last iteration`} />
        <Vital label="Pace" value={recent.length ? `${Math.round(mean(recent))}s` : '–'} sub="per iteration" />
        <Vital label={live ? 'Time left' : 'Run time'} value={live ? duration(remaining * perIteration) : duration(rows.reduce((a, r) => a + r.rollout_seconds + r.update_seconds, 0))} sub={live ? `last update ${ago(data.now - run.updated)}` : 'in total'} />
        <Vital
          label="Latest check"
          value={run.evals.length ? `${(run.evals[run.evals.length - 1].policy * 100).toFixed(1)}%` : '–'}
          sub={run.evals.length ? `rule-based ${(run.evals[run.evals.length - 1].baseline * 100).toFixed(1)}%` : 'none yet'}
          tone={run.evals.length && run.evals[run.evals.length - 1].interval[0] > run.evals[run.evals.length - 1].baseline ? 'good' : ''}
        />
      </div>

      <div className="panel">
        <h3>Charge nurse's alerts</h3>
        <ul className="alerts">
          {alerts(run, data.training, data.now).map((a, i) => <li key={i} className={a.level}>{a.text}</li>)}
        </ul>
      </div>

      <Monitors run={run} />
      <Compare runs={data.runs} />
      <Roster roster={data.roster} />
      <Theatre jobs={data.jobs} load={data.load} cores={data.cores} />
      </>}
    </section>
  )
}
