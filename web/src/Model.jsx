export default function Model({ model }) {
  return (
    <section>
      <p className="muted">
        The learned player (imitation of ladder games, then reinforcement learning) piloting held-out Trick Room teams against
        real teams, compared with a rule-based player piloting the same teams. Each check is 1,200 simulated games, so differences under about 4 points are noise.
      </p>
      {!model.length && <p className="notice">No checks yet.</p>}
      <table>
        <thead><tr><th>Iteration</th><th>Learned player</th><th>Rule-based</th></tr></thead>
        <tbody>
          {model.map((r) => (
            <tr key={r.iteration}>
              <td>{r.iteration}</td>
              <td>{(100 * r.policy).toFixed(1)}% <span className="muted">({(100 * r.interval[0]).toFixed(0)}–{(100 * r.interval[1]).toFixed(0)})</span></td>
              <td>{(100 * r.baseline).toFixed(1)}%</td>
            </tr>
          ))}
        </tbody>
      </table>
    </section>
  )
}
