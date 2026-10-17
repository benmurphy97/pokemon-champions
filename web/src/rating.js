// Lower end of the 95% Wilson interval for wins/games: ranks a team by what we are fairly sure it can do,
// so a team with 70% over 120 games ranks above one with 75% over 8.
export function lowerBound(wins, games, z = 1.96) {
  if (!games) return 0
  const p = wins / games
  const d = 1 + (z * z) / games
  const centre = p + (z * z) / (2 * games)
  const half = z * Math.sqrt((p * (1 - p)) / games + (z * z) / (4 * games * games))
  return (centre - half) / d
}

export const winRate = (r) => `${Math.round((100 * r.wins) / r.games)}%`
