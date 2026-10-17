// Showdown's public sprite host. Hidden if a form has no sprite.
const spriteId = (name) => {
  const [base, ...rest] = name.toLowerCase().split('-')
  return [base, rest.join('')].filter(Boolean).join('-').replace(/[^a-z0-9-]/g, '')
}

export default function Sprite({ name }) {
  return (
    <img
      className="sprite"
      src={`https://play.pokemonshowdown.com/sprites/gen5/${spriteId(name)}.png`}
      alt={name}
      title={name}
      loading="lazy"
      onError={(e) => { e.currentTarget.style.visibility = 'hidden' }}
    />
  )
}
