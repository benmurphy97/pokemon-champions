// Tiny dependency-free SVG line chart for the operating room monitors.
const W = 600

const extent = (values, pad = 0.08) => {
  const lo = Math.min(...values), hi = Math.max(...values)
  const span = hi - lo || 1
  return [lo - span * pad, hi + span * pad]
}

export function rolling(points, window = 5) {
  return points.map(([x], i) => {
    const slice = points.slice(Math.max(0, i - window + 1), i + 1)
    return [x, slice.reduce((a, [, y]) => a + y, 0) / slice.length]
  })
}

/**
 * series: [{ name, color, points: [[x, y]], dots?, dashed? }]
 * band:   { color, points: [[x, lo, hi]] }  (shaded range, e.g. a 95% interval)
 * refs:   [{ y, label, color }]             (horizontal reference lines)
 */
export function Chart({ series, band, refs = [], height = 150, yMin, yMax, format = (v) => v.toFixed(2), label }) {
  const all = series.flatMap((s) => s.points)
  if (!all.length) return <div className="chart-empty">awaiting signal…</div>
  const xs = all.map(([x]) => x)
  const x0 = Math.min(...xs), x1 = Math.max(...xs, x0 + 1)
  const ys = [...all.map(([, y]) => y), ...refs.map((r) => r.y), ...(band ? band.points.flatMap(([, lo, hi]) => [lo, hi]) : [])]
  const [autoLo, autoHi] = extent(ys)
  const lo = yMin ?? autoLo, hi = yMax ?? autoHi
  const padL = 44, padR = 8, padT = 8, padB = 18
  const px = (x) => padL + ((x - x0) / (x1 - x0)) * (W - padL - padR)
  const py = (y) => padT + (1 - (y - lo) / (hi - lo || 1)) * (height - padT - padB)
  const line = (pts) => pts.map(([x, y], i) => `${i ? 'L' : 'M'}${px(x).toFixed(1)},${py(y).toFixed(1)}`).join(' ')
  const ticks = [0, 1, 2, 3].map((i) => lo + ((hi - lo) * i) / 3)

  return (
    <figure className="chart">
      {label && <figcaption>{label}</figcaption>}
      <svg viewBox={`0 0 ${W} ${height}`} role="img" aria-label={label}>
        {ticks.map((t) => (
          <g key={t}>
            <line x1={padL} x2={W - padR} y1={py(t)} y2={py(t)} className="grid" />
            <text x={padL - 6} y={py(t) + 3} className="axis" textAnchor="end">{format(t)}</text>
          </g>
        ))}
        <text x={padL} y={height - 4} className="axis">{Math.round(x0)}</text>
        <text x={W - padR} y={height - 4} className="axis" textAnchor="end">{Math.round(x1)}</text>
        {band && (
          <path
            d={band.points.map(([x, l], i) => `${i ? 'L' : 'M'}${px(x)},${py(l)}`).join(' ') + ' ' +
              [...band.points].reverse().map(([x, , h]) => `L${px(x)},${py(h)}`).join(' ') + ' Z'}
            fill={band.color} opacity="0.16"
          />
        )}
        {refs.map((r) => (
          <g key={r.label}>
            <line x1={padL} x2={W - padR} y1={py(r.y)} y2={py(r.y)} stroke={r.color} strokeDasharray="4 4" opacity="0.8" />
            <text x={W - padR} y={py(r.y) - 3} className="axis" textAnchor="end" fill={r.color}>{r.label}</text>
          </g>
        ))}
        {series.map((s) => (
          <g key={s.name}>
            <path d={line(s.points)} fill="none" stroke={s.color} strokeWidth="2" strokeDasharray={s.dashed ? '5 4' : undefined} strokeLinejoin="round" />
            {s.dots && s.points.map(([x, y]) => <circle key={x} cx={px(x)} cy={py(y)} r="3.5" fill={s.color} />)}
          </g>
        ))}
      </svg>
      <div className="legend">
        {series.map((s) => <span key={s.name}><i style={{ background: s.color }} />{s.name}</span>)}
      </div>
    </figure>
  )
}
