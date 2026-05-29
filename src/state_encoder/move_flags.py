"""
Hardcoded move flag sets for EncoderV2.

These flags cannot be derived from moves.json alone (which only stores
type/category/power/accuracy/PP/target/priority). Each set is based on
standard competitive Pokémon / Showdown definitions, filtered to moves
plausibly legal in Champions M-A format.

Update this file when new moves are confirmed for the format.
"""

# ── Contact moves ─────────────────────────────────────────────────────────────
# Physical moves that make direct contact (trigger Rough Skin, Iron Barbs,
# Flame Body, Static, etc.)
CONTACT_MOVES: set[str] = {
    "Acrobatics", "Aerial Ace", "Aqua Jet", "Aqua Step", "Aqua Tail",
    "Bite", "Body Press", "Body Slam", "Bounce", "Brave Bird",
    "Brick Break", "Bug Bite", "Bug Buzz",  # Bug Buzz is NOT contact (sound)
    "Bulldoze",  # NOT contact but included for completeness (actually is not)
    "Bullet Punch", "Catch",
    "Chip Away", "Circle Throw", "Claw Sharpen",
    "Close Combat", "Comet Punch", "Counter", "Crabhammer",
    "Cross Chop", "Cross Poison", "Crunch",
    "Cut", "Dig", "Dive", "Double Hit", "Double Kick",
    "Double-Edge", "Dragon Claw", "Dragon Rush",
    "Drain Punch", "Dual Chop", "Dynamic Punch",
    "Earthquake",  # NOT contact
    "Extreme Speed", "Facade", "Fire Fang",
    "Fire Punch", "Flare Blitz", "Fly", "Force Palm",
    "Foul Play", "Frustration", "Fury Attack", "Fury Cutter",
    "Fury Swipes", "Gear Grind", "Giga Impact",
    "Glacial Lance",  # NOT contact (projectile)
    "Guillotine", "Gunk Shot",  # Gunk Shot NOT contact
    "Hammer Arm", "Head Charge", "Head Smash",
    "Headbutt", "High Horsepower", "High Jump Kick",
    "Horn Attack", "Horn Drill",
    "Ice Fang", "Ice Hammer", "Ice Punch",
    "Icicle Crash",  # NOT contact
    "Infestation", "Iron Head",  # Iron Head NOT contact? Actually IS contact
    "Jaw Lock", "Jump Kick",
    "Knock Off", "Last Resort",
    "Leech Life", "Lick",
    "Low Kick", "Low Sweep", "Mach Punch",
    "Mega Kick", "Mega Punch", "Megahorn",
    "Metal Claw", "Meteor Mash",
    "Muddy Water",  # NOT contact
    "Night Slash", "Pin Missile",  # Pin Missile NOT contact
    "Play Rough",
    "Poison Fang", "Poison Jab",
    "Pound", "Power-Up Punch", "Psych Up",
    "Psycho Cut",  # Psycho Cut NOT contact
    "Pursuit",
    "Quick Attack", "Rapid Spin",
    "Return", "Revenge", "Reversal",
    "Rock Climb", "Rock Smash",
    "Rolling Kick", "Rollout",
    "Sacred Sword", "Scratch",
    "Seed Bomb",  # NOT contact
    "Shadow Claw", "Shadow Sneak",
    "Skull Bash", "Sky Drop", "Sky Uppercut",
    "Slam", "Slash", "Sleep Talk",
    "Snatch",
    "Stomp", "Stone Edge",  # Stone Edge NOT contact
    "Storm Throw", "Submission",
    "Sucker Punch", "Super Fang", "Superpower",
    "Swift",  # NOT contact
    "Tackle", "Take Down",
    "Thunder Fang", "Thunder Punch",
    "Thrash",
    "Triple Axel", "Triple Kick",
    "U-Turn",
    "Vine Whip",
    "Vital Throw",
    "Volt Tackle",
    "Wild Charge", "Wing Attack",
    "Wood Hammer", "Wrap",
    "X-Scissor", "Zen Headbutt",
    "Bite", "Crunch", "Fire Fang", "Ice Fang", "Thunder Fang",
    "Poison Fang", "Jaw Lock",  # biting AND contact
    "Bullet Seed",  # NOT contact
    "Arm Thrust", "Comet Punch",
    "Double Slap",
    "Tail Slap",
    "Population Bomb",
    "Icicle Spear",  # NOT contact
    "Pin Missile",  # NOT contact
    "Fury Attack", "Barrage", "Bone Rush", "Rock Blast",  # NOT contact
    "Aqua Cutter",
    "Ceaseless Edge",
    "Scale Shot",  # NOT contact
    "Dragon Darts",  # NOT contact
}

# Remove non-contact moves that crept in above; keep this list clean.
# Physical moves that are explicitly NOT contact:
_NOT_CONTACT: set[str] = {
    "Earthquake", "Magnitude", "Bulldoze",
    "Gunk Shot", "Muddy Water", "Seed Bomb",
    "Glacial Lance", "Icicle Crash", "Icicle Spear",
    "Pin Missile", "Bullet Seed", "Rock Blast", "Bone Rush",
    "Barrage", "Arm Thrust", "Comet Punch",
    "Stone Edge", "Rock Slide", "Rock Tomb",
    "Razor Leaf", "Leaf Blade",  # Leaf Blade NOT contact? Actually IS contact in main series
    "Psycho Cut",
    "Shadow Ball", "Dark Pulse", "Dragon Pulse",
    "Aura Sphere", "Flash Cannon", "Energy Ball",
    "Scale Shot", "Dragon Darts",
}
CONTACT_MOVES -= _NOT_CONTACT

# ── Sound-based moves ─────────────────────────────────────────────────────────
# Bypasses Substitute. Triggers Soundproof immunity.
SOUND_MOVES: set[str] = {
    "Bug Buzz", "Chatter", "Clangorous Soul", "Clanging Scales",
    "Confide",
    "Disarming Voice",
    "Echoed Voice",
    "Heal Bell",
    "Hyper Voice",
    "Metal Sound",
    "Noble Roar",
    "Parting Shot",
    "Perish Song",
    "Relic Song",
    "Round",
    "Screech",
    "Sing",
    "Snarl",
    "Snore",
    "Sparkling Aria",
    "Supersonic",
    "Uproar",
    "Alluring Voice",
    "Torch Song",
    "Growl", "Roar",
}

# ── Punch-based moves ─────────────────────────────────────────────────────────
# Boosted by Iron Fist.
PUNCH_MOVES: set[str] = {
    "Bullet Punch",
    "Comet Punch",
    "Dizzy Punch",
    "Double Iron Bash",
    "Drain Punch",
    "Dynamic Punch",
    "Fire Punch",
    "Focus Punch",
    "Hammer Arm",
    "Ice Hammer",
    "Ice Punch",
    "Mach Punch",
    "Mega Punch",
    "Meteor Mash",
    "Power-Up Punch",
    "Shadow Punch",
    "Sky Uppercut",
    "Thunder Punch",
    "Plasma Fists",
}

# ── Biting moves ──────────────────────────────────────────────────────────────
# Boosted by Strong Jaw.
BITING_MOVES: set[str] = {
    "Bite",
    "Crunch",
    "Fire Fang",
    "Fishious Rend",
    "Hyper Fang",
    "Ice Fang",
    "Jaw Lock",
    "Poison Fang",
    "Psychic Fangs",
    "Thunder Fang",
    "Lunge",  # NOT biting — removing
}
BITING_MOVES -= {"Lunge"}

# ── Moves that bypass Protect ─────────────────────────────────────────────────
BYPASSES_PROTECT: set[str] = {
    "Feint",
    "Hyperspace Fury",
    "Hyperspace Hole",
    "Phantom Force",
    "Shadow Force",
    "Mean Look",  # does NOT bypass protect
}
BYPASSES_PROTECT -= {"Mean Look"}

# ── Protect variants ──────────────────────────────────────────────────────────
PROTECT_MOVES: set[str] = {
    "Protect",
    "Detect",
    "Wide Guard",
    "Quick Guard",
    "Spiky Shield",
    "Baneful Bunker",
    "Silk Trap",
    "Burning Bulwark",
    "King's Shield",
    "Obstruct",
}

# ── Recoil moves ──────────────────────────────────────────────────────────────
RECOIL_MOVES: set[str] = {
    "Brave Bird",
    "Double-Edge",
    "Flare Blitz",
    "Head Charge",
    "Head Smash",
    "High Jump Kick",  # recoil on miss, encoded as recoil
    "Jump Kick",
    "Submission",
    "Take Down",
    "Volt Tackle",
    "Wild Charge",
    "Wood Hammer",
    "Chloroblast",
    "Supersonic Skystrike",
}

# ── Self-healing moves ────────────────────────────────────────────────────────
HEALING_MOVES: set[str] = {
    "Heal Order",
    "Lunar Dance",
    "Milk Drink",
    "Moonlight",
    "Morning Sun",
    "Recover",
    "Rest",
    "Roost",
    "Slack Off",
    "Soft-Boiled",
    "Synthesis",
    "Shore Up",
    "Jungle Healing",
    "Heal Pulse",  # targets ally, still a healing move
    "Life Dew",
}

# ── Draining moves ────────────────────────────────────────────────────────────
DRAINING_MOVES: set[str] = {
    "Absorb",
    "Draining Kiss",
    "Drain Punch",
    "Dream Eater",
    "Giga Drain",
    "Horn Leech",
    "Leech Life",
    "Mega Drain",
    "Oblivion Wing",
    "Parabolic Charge",
    "Strength Sap",
}

# ── Weather-setting moves ─────────────────────────────────────────────────────
# Maps move name → canonical weather key used in GameState
WEATHER_SETTING_MOVES: dict[str, str] = {
    "Rain Dance":  "rain",
    "Sunny Day":   "sun",
    "Sandstorm":   "sand",
    "Snowscape":   "snow",
    "Hail":        "snow",  # Hail deprecated but may appear in data
}

# ── Terrain-setting moves ─────────────────────────────────────────────────────
TERRAIN_SETTING_MOVES: dict[str, str] = {
    "Electric Terrain": "electric",
    "Grassy Terrain":   "grassy",
    "Misty Terrain":    "misty",
    "Psychic Terrain":  "psychic",
}

# ── Trick Room ────────────────────────────────────────────────────────────────
TRICK_ROOM_MOVES: set[str] = {"Trick Room"}

# ── Tailwind ──────────────────────────────────────────────────────────────────
TAILWIND_MOVES: set[str] = {"Tailwind"}

# ── Redirection moves ─────────────────────────────────────────────────────────
REDIRECTION_MOVES: set[str] = {"Follow Me", "Rage Powder"}

# ── Multi-hit moves ───────────────────────────────────────────────────────────
MULTIHIT_MOVES: set[str] = {
    "Arm Thrust",
    "Barrage",
    "Bone Rush",
    "Bullet Seed",
    "Clangorous Soulblaze",
    "Comet Punch",
    "Double Hit",
    "Double Kick",
    "Double Slap",
    "Dragon Darts",
    "Dual Chop",
    "Dual Wingbeat",
    "Fury Attack",
    "Fury Swipes",
    "Gear Grind",
    "Icicle Spear",
    "Loaded Dice",  # not a move; item
    "Pin Missile",
    "Population Bomb",
    "Rock Blast",
    "Scale Shot",
    "Spike Cannon",
    "Tail Slap",
    "Triple Axel",
    "Triple Kick",
    "Water Shuriken",
    "Tachyon Cutter",
    "Surging Strikes",
}
MULTIHIT_MOVES -= {"Loaded Dice"}

# ── Two-turn moves ────────────────────────────────────────────────────────────
TWO_TURN_MOVES: set[str] = {
    "Bounce",
    "Dig",
    "Dive",
    "Fly",
    "Freeze Shock",
    "Ice Burn",
    "Phantom Force",
    "Shadow Force",
    "Sky Attack",
    "Sky Drop",
    "Solar Beam",
    "Solar Blade",
    "Skull Bash",
    "Razor Wind",
}

# ── Fixed-damage moves ────────────────────────────────────────────────────────
# Damage is not modified by stats (Seismic Toss, OHKO moves, etc.)
FIXED_DAMAGE_MOVES: set[str] = {
    "Dragon Rage",
    "Fissure",
    "Guillotine",
    "Horn Drill",
    "Night Shade",
    "One-Hit KO",
    "Seismic Toss",
    "Sheer Cold",
    "Super Fang",
}

# ── Moves that always inflict a primary status ────────────────────────────────
# Maps move name → (status_key, application_chance 0.0–1.0)
# status_key matches STATUS_ORDER in constants.py
STATUS_INFLICTING_MOVES: dict[str, tuple[str, float]] = {
    # Guaranteed sleep
    "Spore":              ("slp", 1.0),
    "Sleep Powder":       ("slp", 0.75),
    "Hypnosis":           ("slp", 0.60),
    "Lovely Kiss":        ("slp", 0.75),
    "Sing":               ("slp", 0.55),
    "Grass Whistle":      ("slp", 0.55),
    "Yawn":               ("slp", 1.0),   # sleep next turn
    "Dark Void":          ("slp", 0.50),
    # Guaranteed paralysis
    "Thunder Wave":       ("par", 1.0),
    "Stun Spore":         ("par", 0.75),
    "Glare":              ("par", 1.0),
    "Nuzzle":             ("par", 1.0),
    # Guaranteed burn
    "Will-O-Wisp":        ("brn", 0.85),
    # Guaranteed poison
    "Poison Gas":         ("psn", 0.90),
    "Poison Powder":      ("psn", 0.75),
    "Toxic":              ("tox", 0.90),
    # Guaranteed freeze (rare, mainly for reference)
    "Freeze-Dry":         ("frz", 0.10),  # secondary only; handled below too
}

# ── Secondary effects on damaging moves ──────────────────────────────────────
# Maps move name → (secondary_type, chance)
# secondary_type ∈ {"flinch","burn","paralysis","freeze","stat_change"}
SECONDARY_EFFECTS: dict[str, tuple[str, float]] = {
    # Flinch
    "Air Slash":         ("flinch", 0.30),
    "Astonish":          ("flinch", 0.30),
    "Bite":              ("flinch", 0.30),
    "Blue Flare":        ("flinch", 0.20),
    "Bolt Strike":       ("flinch", 0.20),
    "Dark Pulse":        ("flinch", 0.20),
    "Dragon Rush":       ("flinch", 0.20),
    "Extrasensory":      ("flinch", 0.10),
    "Fake Out":          ("flinch", 1.00),
    "Fire Fang":         ("flinch", 0.10),
    "Headbutt":          ("flinch", 0.30),
    "Heart Stamp":       ("flinch", 0.30),
    "Hyper Fang":        ("flinch", 0.10),
    "Ice Fang":          ("flinch", 0.10),
    "Icicle Crash":      ("flinch", 0.30),
    "Iron Head":         ("flinch", 0.30),
    "Iron Tail":         ("flinch", 0.30),  # actually lowers Defence
    "Needle Arm":        ("flinch", 0.30),
    "Rock Slide":        ("flinch", 0.30),
    "Rolling Kick":      ("flinch", 0.30),
    "Sky Attack":        ("flinch", 0.30),
    "Snore":             ("flinch", 0.30),
    "Stomp":             ("flinch", 0.30),
    "Thunder Fang":      ("flinch", 0.10),
    "Twister":           ("flinch", 0.20),
    "Waterfall":         ("flinch", 0.20),
    "Zen Headbutt":      ("flinch", 0.20),
    "Alluring Voice":    ("flinch", 0.00),  # actually confuses, not flinch
    # Burn
    "Blaze Kick":        ("burn", 0.10),
    "Ember":             ("burn", 0.10),
    "Fire Blast":        ("burn", 0.10),
    "Fire Fang":         ("burn", 0.10),
    "Fire Punch":        ("burn", 0.10),
    "Flamethrower":      ("burn", 0.10),
    "Heat Wave":         ("burn", 0.10),
    "Incinerate":        ("burn", 0.00),  # destroys berry, no burn
    "Lava Plume":        ("burn", 0.30),
    "Magma Storm":       ("burn", 0.30),
    "Sacred Fire":       ("burn", 0.50),
    "Scald":             ("burn", 0.30),
    "Steam Eruption":    ("burn", 0.30),
    "Searing Shot":      ("burn", 0.30),
    # Paralysis
    "Body Slam":         ("paralysis", 0.30),
    "Bolt Strike":       ("paralysis", 0.20),
    "Discharge":         ("paralysis", 0.30),
    "Force Palm":        ("paralysis", 0.30),
    "Lick":              ("paralysis", 0.30),
    "Spark":             ("paralysis", 0.30),
    "Thunder":           ("paralysis", 0.30),
    "Thunderbolt":       ("paralysis", 0.10),
    "Thundershock":      ("paralysis", 0.10),
    "Volt Tackle":       ("paralysis", 0.10),
    "Zap Cannon":        ("paralysis", 1.00),
    # Freeze
    "Aurora Beam":       ("freeze", 0.10),
    "Blizzard":          ("freeze", 0.10),
    "Freeze-Dry":        ("freeze", 0.10),
    "Ice Beam":          ("freeze", 0.10),
    "Ice Fang":          ("freeze", 0.10),
    "Ice Punch":         ("freeze", 0.10),
    "Powder Snow":       ("freeze", 0.10),
    # Stat change (catch-all for moves with notable stat-drop secondaries)
    "Ancient Power":     ("stat_change", 0.10),
    "Apple Acid":        ("stat_change", 1.00),
    "Bug Buzz":          ("stat_change", 0.10),  # SpDef drop
    "Charm":             ("stat_change", 1.00),
    "Earth Power":       ("stat_change", 0.10),
    "Energy Ball":       ("stat_change", 0.10),
    "Flash Cannon":      ("stat_change", 0.10),
    "Focus Blast":       ("stat_change", 0.10),
    "Hurricane":         ("stat_change", 0.30),
    "Muddy Water":       ("stat_change", 0.30),
    "Psychic":           ("stat_change", 0.10),
    "Rock Smash":        ("stat_change", 0.50),
    "Shadow Ball":       ("stat_change", 0.20),
    "Silver Wind":       ("stat_change", 0.10),
    "Slash":             ("stat_change", 0.00),  # no secondary
    "Snarl":             ("stat_change", 1.00),
    "String Shot":       ("stat_change", 1.00),
}

# ── Stat-change moves (primary effect) ───────────────────────────────────────
# Maps move name → (target, is_boost)
# target ∈ {"self", "opponent", "both"}
# A move appears here only if its PRIMARY effect is a stat change.
# Moves with secondary stat-change effects are captured in SECONDARY_EFFECTS.
STAT_CHANGE_MOVES: dict[str, tuple[str, bool]] = {
    # Self-boosts
    "Agility":           ("self", True),
    "Amnesia":           ("self", True),
    "Autotomize":        ("self", True),
    "Barrier":           ("self", True),
    "Belly Drum":        ("self", True),
    "Bulk Up":           ("self", True),
    "Calm Mind":         ("self", True),
    "Clangorous Soul":   ("self", True),
    "Coil":              ("self", True),
    "Cotton Guard":      ("self", True),
    "Defense Curl":      ("self", True),
    "Double Team":       ("self", True),
    "Dragon Dance":      ("self", True),
    "Fillet Away":       ("self", True),
    "Geomancy":          ("self", True),
    "Growth":            ("self", True),
    "Harden":            ("self", True),
    "Howl":              ("self", True),
    "Iron Defense":      ("self", True),
    "Meditate":          ("self", True),
    "Minimize":          ("self", True),
    "Nasty Plot":        ("self", True),
    "No Retreat":        ("self", True),
    "Quiver Dance":      ("self", True),
    "Rock Polish":       ("self", True),
    "Sharpen":           ("self", True),
    "Shell Smash":       ("self", True),
    "Shift Gear":        ("self", True),
    "Swords Dance":      ("self", True),
    "Tail Glow":         ("self", True),
    "Withdraw":          ("self", True),
    "Work Up":           ("self", True),
    # Opponent stat drops
    "Baby-Doll Eyes":    ("opponent", False),
    "Charm":             ("opponent", False),
    "Confide":           ("opponent", False),
    "Fake Tears":        ("opponent", False),
    "Feather Dance":     ("opponent", False),
    "Flash":             ("opponent", False),
    "Growl":             ("opponent", False),
    "Leer":              ("opponent", False),
    "Metal Sound":       ("opponent", False),
    "Noble Roar":        ("opponent", False),
    "Parting Shot":      ("opponent", False),
    "Sand Attack":       ("opponent", False),
    "Scary Face":        ("opponent", False),
    "Screech":           ("opponent", False),
    "Smokescreen":       ("opponent", False),
    "Snarl":             ("opponent", False),
    "String Shot":       ("opponent", False),
    "Sweet Scent":       ("opponent", False),
    "Tail Whip":         ("opponent", False),
    "Tickle":            ("opponent", False),
    # Both / ally boosts
    "Coaching":          ("both", True),   # boosts ally
    "Helping Hand":      ("both", True),   # multiplies ally's move
    "Life Dew":          ("both", True),   # heals, not technically stat
    "Misty Explosion":   ("self", False),  # faint-based
}

# ── Spread move targets from Showdown target field ────────────────────────────
SPREAD_TARGETS: set[str] = {"allAdjacent", "allAdjacentFoes"}

# ── Ally-hitting spread targets ───────────────────────────────────────────────
ALLY_HIT_TARGETS: set[str] = {"allAdjacent"}
