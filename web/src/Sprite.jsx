// Showdown's public sprite host. `animated` uses the animated sprites (front or back view), falling back to the
// static one, then to the base species (new Megas have no sprite yet), then hiding.
const spriteId = (name) => {
  const [base, ...rest] = name.toLowerCase().split('-')
  return [base, rest.join('')].filter(Boolean).join('-').replace(/[^a-z0-9-]/g, '')
}
const HOST = 'https://play.pokemonshowdown.com/sprites/'

export default function Sprite({ name, animated, back }) {
  const id = spriteId(name)
  const first = animated ? `${HOST}${back ? 'ani-back' : 'ani'}/${id}.${'gif'}` : `${HOST}gen5/${id}.png`
  const fallback = (e) => {
    const img = e.currentTarget
    const step = Number(img.dataset.step || 0)
    img.dataset.step = step + 1
    if (animated && step === 0) img.src = `${HOST}${back ? 'gen5-back' : 'gen5'}/${id}.png`
    else if (animated && step === 1) img.src = `${HOST}${back ? 'ani-back' : 'ani'}/${id.split('-')[0]}.gif`
    else img.style.visibility = 'hidden'
  }
  return <img className={animated ? 'sprite big' : 'sprite'} src={first} alt={name} title={name} loading="lazy" onError={fallback} />
}
