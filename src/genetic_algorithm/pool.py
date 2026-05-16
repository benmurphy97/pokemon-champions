"""
Curated pool of meta-relevant Pokemon templates for team mutation/swapping.
Each entry provides enough info to slot a Pokemon into a valid Showdown team string.

Only includes Pokemon verified available in gen9championsvgc2026regma.
Mega stones are only listed for Megas confirmed working on our local Showdown server.
"""

from dataclasses import dataclass, field


@dataclass
class PokemonTemplate:
    name: str
    ability: str
    items: list[str]          # ordered by preference
    moves: list[str]          # ≥4 candidates; first 4 used as default
    evs: dict[str, int]       # stat → EV (total must be ≤66, each ≤32)
    nature: str
    role: str                 # human-readable archetype tag


# ---------------------------------------------------------------------------
# Support / utility
# ---------------------------------------------------------------------------
POOL: list[PokemonTemplate] = [

    PokemonTemplate(
        name="Incineroar",
        ability="Intimidate",
        items=["Sitrus Berry", "Chople Berry", "Lum Berry"],
        moves=["Fake Out", "Parting Shot", "Throat Chop", "Flare Blitz",
               "Knock Off", "Will-O-Wisp", "Darkest Lariat", "Protect"],
        evs={"HP": 32, "Def": 14, "SpD": 20},
        nature="Careful",
        role="support",
    ),

    PokemonTemplate(
        name="Farigiraf",
        ability="Armor Tail",
        items=["Mental Herb", "Sitrus Berry"],
        moves=["Trick Room", "Helping Hand", "Hyper Voice", "Psychic",
               "Expanding Force", "Future Sight", "Dazzling Gleam", "Protect"],
        evs={"HP": 32, "Def": 2, "SpA": 32},
        nature="Quiet",
        role="trick-room setter",
    ),

    PokemonTemplate(
        name="Sinistcha",
        ability="Hospitality",
        items=["Sitrus Berry", "Kasib Berry", "Coba Berry"],
        moves=["Trick Room", "Matcha Gotcha", "Rage Powder", "Life Dew",
               "Protect", "Helping Hand"],
        evs={"HP": 32, "Def": 10, "SpD": 24},
        nature="Calm",
        role="trick-room setter / support",
    ),

    PokemonTemplate(
        name="Hatterene",
        ability="Magic Bounce",
        items=["Fairy Feather", "Mental Herb", "Focus Sash"],
        moves=["Trick Room", "Psychic", "Dazzling Gleam", "Protect",
               "Mystical Fire", "Heal Pulse"],
        evs={"HP": 32, "Def": 2, "SpA": 32},
        nature="Quiet",
        role="trick-room setter",
    ),

    PokemonTemplate(
        name="Oranguru",
        ability="Inner Focus",
        items=["Mental Herb", "Sitrus Berry"],
        moves=["Trick Room", "Instruct", "Psychic", "Protect",
               "Helping Hand", "Hypnosis"],
        evs={"HP": 32, "Def": 20, "SpD": 14},
        nature="Quiet",
        role="trick-room setter / instruct",
    ),

    PokemonTemplate(
        name="Pelipper",
        ability="Drizzle",
        items=["Choice Scarf", "Sitrus Berry", "Lum Berry"],
        moves=["Hurricane", "Weather Ball", "U-turn", "Tailwind",
               "Wide Guard", "Protect"],
        evs={"HP": 2, "SpA": 32, "Spe": 32},
        nature="Timid",
        role="rain setter",
    ),

    PokemonTemplate(
        name="Torkoal",
        ability="Drought",
        items=["Charcoal", "Sitrus Berry", "Mental Herb"],
        moves=["Eruption", "Heat Wave", "Protect", "Clear Smog",
               "Earth Power", "Will-O-Wisp"],
        evs={"HP": 32, "SpA": 32, "SpD": 2},
        nature="Quiet",
        role="sun setter",
    ),

    PokemonTemplate(
        name="Ninetales-Alola",
        ability="Snow Warning",
        items=["Focus Sash", "Never-Melt Ice"],
        moves=["Freeze-Dry", "Blizzard", "Aurora Veil", "Protect",
               "Dazzling Gleam", "Moonblast"],
        evs={"HP": 2, "SpA": 32, "Spe": 32},
        nature="Timid",
        role="snow setter",
    ),

    # ---------------------------------------------------------------------------
    # Physical sweepers / attackers
    # ---------------------------------------------------------------------------

    PokemonTemplate(
        name="Sneasler",
        ability="Unburden",
        items=["White Herb", "Focus Sash"],
        moves=["Fake Out", "Close Combat", "Dire Claw", "Protect",
               "Rock Slide", "Poison Jab"],
        evs={"Atk": 32, "SpD": 2, "Spe": 32},
        nature="Adamant",
        role="lead pressure",
    ),

    PokemonTemplate(
        name="Garchomp",
        ability="Rough Skin",
        items=["Choice Scarf", "Lum Berry", "Dragon Fang"],
        moves=["Earthquake", "Dragon Claw", "Rock Slide", "Protect",
               "Bulldoze", "Iron Head"],
        evs={"HP": 2, "Atk": 32, "Spe": 32},
        nature="Jolly",
        role="scarf sweeper",
    ),

    PokemonTemplate(
        name="Kingambit",
        ability="Defiant",
        items=["Chople Berry", "Black Glasses", "Lum Berry"],
        moves=["Kowtow Cleave", "Iron Head", "Sucker Punch", "Protect",
               "Low Kick", "Brick Break"],
        evs={"HP": 32, "Atk": 32, "Def": 2},
        nature="Adamant",
        role="endgame cleaner",
    ),

    PokemonTemplate(
        name="Kleavor",
        ability="Sheer Force",
        items=["Sharp Beak", "Focus Sash", "Hard Stone"],
        moves=["Stone Axe", "X-Scissor", "Close Combat", "Protect",
               "Rock Slide", "Aerial Ace"],
        evs={"HP": 2, "Atk": 32, "Spe": 32},
        nature="Jolly",
        role="spread + cleave",
    ),

    PokemonTemplate(
        name="Basculegion",
        ability="Adaptability",
        items=["Choice Scarf", "Mystic Water"],
        moves=["Last Respects", "Wave Crash", "Aqua Jet", "Flip Turn",
               "Ice Spinner"],
        evs={"HP": 2, "Atk": 32, "Spe": 32},
        nature="Adamant",
        role="rain nuke / revenge",
    ),

    PokemonTemplate(
        name="Aerodactyl",
        ability="Unnerve",
        items=["Focus Sash", "Choice Scarf"],
        moves=["Rock Slide", "Dual Wingbeat", "Tailwind", "Protect",
               "Earthquake", "Ice Fang"],
        evs={"HP": 2, "Atk": 32, "Spe": 32},
        nature="Jolly",
        role="tailwind setter / speed control",
    ),

    PokemonTemplate(
        name="Tyranitar",
        ability="Sand Stream",
        items=["Choice Scarf", "Chople Berry"],
        moves=["Rock Slide", "Crunch", "Ice Punch", "Protect",
               "Low Kick", "Knock Off"],
        evs={"HP": 2, "Atk": 32, "Spe": 32},
        nature="Jolly",
        role="sand setter / physical attacker",
    ),

    PokemonTemplate(
        name="Talonflame",
        ability="Gale Wings",
        items=["Focus Sash", "Sitrus Berry"],
        moves=["Tailwind", "Brave Bird", "Flare Blitz", "Protect",
               "Quick Attack", "Will-O-Wisp"],
        evs={"HP": 2, "Atk": 32, "Spe": 32},
        nature="Jolly",
        role="tailwind setter / priority",
    ),

    PokemonTemplate(
        name="Whimsicott",
        ability="Prankster",
        items=["Focus Sash", "Lum Berry", "Mental Herb"],
        moves=["Tailwind", "Encore", "Fake Tears", "Protect",
               "Helping Hand", "Beat Up"],
        evs={"HP": 2, "SpA": 32, "Spe": 32},
        nature="Timid",
        role="priority support / tailwind",
    ),

    PokemonTemplate(
        name="Archaludon",
        ability="Stamina",
        items=["Choice Scarf", "Sitrus Berry"],
        moves=["Electro Shot", "Dragon Pulse", "Flash Cannon", "Protect",
               "Body Press", "Draco Meteor"],
        evs={"HP": 2, "SpA": 32, "Spe": 32},
        nature="Timid",
        role="electric / dragon attacker",
    ),

    PokemonTemplate(
        name="Sylveon",
        ability="Pixilate",
        items=["Sitrus Berry", "Fairy Feather"],
        moves=["Hyper Voice", "Moonblast", "Helping Hand", "Protect",
               "Yawn", "Mystical Fire"],
        evs={"HP": 32, "SpA": 2, "SpD": 32},
        nature="Calm",
        role="fairy spread / bulk",
    ),

    # ---------------------------------------------------------------------------
    # Special attackers
    # ---------------------------------------------------------------------------

    PokemonTemplate(
        name="Venusaur",
        ability="Chlorophyll",
        items=["Focus Sash", "Miracle Seed"],
        moves=["Sludge Bomb", "Sleep Powder", "Protect", "Leaf Storm",
               "Giga Drain", "Energy Ball", "Weather Ball"],
        evs={"HP": 2, "SpA": 32, "Spe": 32},
        nature="Modest",
        role="sun sweeper",
    ),

    PokemonTemplate(
        name="Blastoise",
        ability="Rain Dish",
        items=["Mystic Water", "Sitrus Berry"],
        moves=["Water Spout", "Fake Out", "Flip Turn", "Protect",
               "Ice Beam", "Muddy Water", "Yawn"],
        evs={"HP": 18, "SpA": 32, "Spe": 16},
        nature="Modest",
        role="rain attacker / pivot",
    ),

    PokemonTemplate(
        name="Rotom-Wash",
        ability="Levitate",
        items=["Sitrus Berry", "Leftovers"],
        moves=["Hydro Pump", "Thunderbolt", "Will-O-Wisp", "Protect",
               "Volt Switch", "Icy Wind"],
        evs={"HP": 32, "SpA": 16, "SpD": 18},
        nature="Bold",
        role="bulky pivot",
    ),

    PokemonTemplate(
        name="Rotom-Heat",
        ability="Levitate",
        items=["Sitrus Berry", "Choice Scarf"],
        moves=["Overheat", "Thunderbolt", "Helping Hand", "Protect",
               "Volt Switch", "Will-O-Wisp"],
        evs={"HP": 32, "SpA": 32, "Spe": 2},
        nature="Modest",
        role="sun support / volt turn",
    ),

    PokemonTemplate(
        name="Aegislash",
        ability="Stance Change",
        items=["Leftovers", "Chople Berry"],
        moves=["Shadow Ball", "Flash Cannon", "Wide Guard", "King's Shield",
               "Close Combat", "Shadow Sneak"],
        evs={"HP": 32, "SpA": 32, "SpD": 2},
        nature="Quiet",
        role="wide guard / spread attacker",
    ),

    PokemonTemplate(
        name="Hydreigon",
        ability="Levitate",
        items=["Choice Scarf", "Dragon Fang"],
        moves=["Draco Meteor", "Dark Pulse", "Heat Wave", "Protect",
               "Flash Cannon", "Earth Power"],
        evs={"HP": 2, "SpA": 32, "Spe": 32},
        nature="Timid",
        role="scarfer / coverage",
    ),

    PokemonTemplate(
        name="Milotic",
        ability="Competitive",
        items=["Leftovers", "Sitrus Berry"],
        moves=["Muddy Water", "Icy Wind", "Recover", "Protect",
               "Scald", "Hypnosis", "Coil"],
        evs={"HP": 32, "Def": 22, "SpD": 12},
        nature="Calm",
        role="bulk / intimidate punisher",
    ),

    PokemonTemplate(
        name="Clefable",
        ability="Unaware",
        items=["Focus Sash", "Sitrus Berry"],
        moves=["Moonblast", "Helping Hand", "Follow Me", "Protect",
               "Charm", "Encore"],
        evs={"HP": 32, "SpA": 32, "SpD": 2},
        nature="Modest",
        role="redirection / unaware",
    ),

    # ---------------------------------------------------------------------------
    # Mega Evolutions (verified on local Showdown)
    # ---------------------------------------------------------------------------

    PokemonTemplate(
        name="Charizard-Mega-Y",
        ability="Solar Power",
        items=["Charizardite Y"],
        moves=["Heat Wave", "Solar Beam", "Weather Ball", "Protect",
               "Ancient Power", "Air Slash", "Fire Blast"],
        evs={"HP": 2, "SpA": 32, "Spe": 32},
        nature="Timid",
        role="sun nuke",
    ),

    PokemonTemplate(
        name="Charizard-Mega-X",
        ability="Tough Claws",
        items=["Charizardite X"],
        moves=["Flare Blitz", "Dragon Claw", "Dragon Dance", "Protect",
               "Earthquake", "Aerial Ace"],
        evs={"HP": 14, "Atk": 21, "Spe": 31},
        nature="Adamant",
        role="setup sweeper",
    ),

    PokemonTemplate(
        name="Tyranitar-Mega",
        ability="Sand Stream",
        items=["Tyranitarite"],
        moves=["Rock Slide", "Crunch", "Protect", "High Horsepower",
               "Ice Punch", "Low Kick"],
        evs={"HP": 17, "Atk": 26, "Spe": 23},
        nature="Adamant",
        role="sand attacker",
    ),

    PokemonTemplate(
        name="Blastoise-Mega",
        ability="Mega Launcher",
        items=["Blastoisinite"],
        moves=["Water Spout", "Water Pulse", "Ice Beam", "Protect",
               "Aura Sphere", "Dark Pulse"],
        evs={"HP": 2, "SpA": 32, "Spe": 32},
        nature="Timid",
        role="rain nuke",
    ),

    PokemonTemplate(
        name="Gengar-Mega",
        ability="Shadow Tag",
        items=["Gengarite"],
        moves=["Shadow Ball", "Protect", "Substitute", "Perish Song",
               "Focus Blast", "Sludge Bomb"],
        evs={"HP": 32, "SpA": 2, "Spe": 32},
        nature="Timid",
        role="perish trap",
    ),

    PokemonTemplate(
        name="Golurk-Mega",
        ability="Iron Fist",
        items=["Golurkite"],
        moves=["Headlong Rush", "Thunder Punch", "Drain Punch", "Protect",
               "Shadow Punch", "Ice Punch"],
        evs={"HP": 32, "Atk": 32, "Def": 2},
        nature="Brave",
        role="trick-room physical attacker",
    ),

    PokemonTemplate(
        name="Starmie-Mega",
        ability="Natural Cure",
        items=["Starminite"],
        moves=["Liquidation", "Zen Headbutt", "Skill Swap", "Protect",
               "Ice Beam", "Thunderbolt"],
        evs={"HP": 2, "Atk": 32, "Spe": 32},
        nature="Jolly",
        role="fast physical attacker",
    ),

    PokemonTemplate(
        name="Froslass-Mega",
        ability="Snow Warning",
        items=["Froslassite"],
        moves=["Blizzard", "Shadow Ball", "Aurora Veil", "Protect",
               "Icy Wind", "Destiny Bond"],
        evs={"HP": 2, "SpA": 32, "Spe": 32},
        nature="Timid",
        role="snow setter / veil",
    ),

    PokemonTemplate(
        name="Aerodactyl-Mega",
        ability="Tough Claws",
        items=["Aerodactylite"],
        moves=["Rock Slide", "Tailwind", "Protect", "Earthquake",
               "Ice Fang", "Aerial Ace"],
        evs={"HP": 2, "Atk": 32, "Spe": 32},
        nature="Jolly",
        role="fast physical attacker / tailwind",
    ),

    PokemonTemplate(
        name="Floette-Eternal",
        ability="Flower Veil",
        items=["Floettite"],
        moves=["Moonblast", "Dazzling Gleam", "Protect", "Aromatherapy",
               "Light Screen", "Helping Hand"],
        evs={"HP": 32, "SpA": 2, "SpD": 32},
        nature="Calm",
        role="fairy bulk / support",
    ),
]

# Quick lookup by name
POOL_BY_NAME: dict[str, PokemonTemplate] = {p.name: p for p in POOL}

# Mega Stone items — locked to specific Pokemon and never swapped to others
MEGA_STONES: set[str] = {
    p.items[0] for p in POOL if p.items and p.items[0].endswith("ite")
}
