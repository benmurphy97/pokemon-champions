# Stage 3: In-Battle State Encoding — v2 Full Feature List
**Pokémon VGC AI System — In-Battle Decision Module**
**Format: Pokémon Champions — Regulation Set M-A (Masters Division)**

---

## Format Reference: Regulation Set M-A

This encoding is designed specifically for **Pokémon Champions Regulation Set M-A**, the current competitive format as of May 2026. Key format facts that directly affect the encoding:

| Rule | Detail |
|---|---|
| Game | Pokémon Champions (Nintendo Switch, launched April 8 2026) |
| Format | Doubles — 2v2, 6 Pokémon registered, choose 4, lead 2 |
| Level cap | All Pokémon auto-levelled to 50 |
| Stat system | SP (Stat Points) replaces EVs/IVs — 66 SP per Pokémon, max 32 per stat |
| Nature system | Stat Alignment replaces Natures — same +10%/-10% mechanic, neutral = Serious only |
| Mega Evolution | Allowed — one Mega per battle. Mega changes ability, sometimes type, immediately on selection |
| Legendaries | Banned — no Restricted Pokémon, no Paradox Pokémon |
| Mythicals | Banned |
| Pokémon pool | ~263 Pokémon available in Regulation M-A roster |
| Item clause | No duplicate items on the same team |
| Species clause | No duplicate species on the same team |
| Team preview | Full Open Team Sheet — both players see all 6 opponents before selecting 4 |
| Terastallization | Not present in Pokémon Champions — does NOT apply to this format |
| IVs | Removed in Pokémon Champions — all Pokémon have perfect stats by default |
| Duration | April 8 – June 17, 2026 |

**Critical encoding implications:**
- No IVs to encode — removed entirely from the game
- SP replaces EVs — 66 points, max 32 per stat. Encode SP not EVs
- Stat Alignment replaces Nature — same +10%/-10% modifier, same encoding approach
- Mega Evolution is a mid-battle transformation — requires a transformation event in the state
- Terastallization does not exist in this format — remove all Tera-related encoding
- No Legendaries or Paradox Pokémon — format filter blocks these entirely
- Only one Mega Evolution allowed per battle — track whether Mega has been used

---

## Format Legality Filter

A **format legality filter** sits upstream of the encoder. The encoder never sees illegal content. The filter validates:

- Pokémon species is in the M-A legal roster (~263 Pokémon)
- No duplicate species on the team (Species Clause)
- No duplicate items on the team (Item Clause)
- No Legendary, Mythical, or Paradox Pokémon
- Mega Stones are legal for Pokémon with available Mega Evolutions in M-A
- Moves are in the Pokémon's legal learnset for Champions

The filter is a standalone module that loads the current format's legal list. When the format changes (M-B, M-C etc.) only the filter updates — the encoder architecture is unchanged.

---

## Design Principles

- **Encode effects, not identities.** Moves and abilities described by what they do, not by name or ID. Enables generalisation to new moves and abilities.
- **Normalise all continuous values to 0–1.** Divide by the true maximum for each stat.
- **Represent uncertainty explicitly.** Opponent unknowns use a dedicated unknown/prior flag. The model must distinguish "zero attack boosts" from "unknown item".
- **Encode constraints.** Action legality (choice lock, taunt, encore, no PP) is in the state so the model never wastes capacity learning that illegal actions are bad. Invalid actions masked at the output layer.
- **Separate structure from content.** Every turn the encoding has the same shape. Fainted or unrevealed slots are zero-padded with explicit "slot empty / not yet seen" flags.
- **No species or move IDs.** No one-hot over 263 species or hundreds of moves. Every Pokémon and move described by its properties.
- **SP not EVs.** 66 SP total, max 32 per stat. Normalise SP ÷ 32 per stat.
- **No IVs.** Removed from the game entirely. Do not encode.
- **No Terastallization.** Does not exist in this format. Not encoded.

---

## Stat Normalisation Reference

| Stat | Max Possible at Level 50 | Divide By |
|---|---|---|
| HP (actual) | ~400 (high HP base + 32 SP) | 400 |
| Attack | ~350 (high Atk base + 32 SP + Alignment) | 350 |
| Defence | ~350 | 350 |
| Special Attack | ~350 | 350 |
| Special Defence | ~350 | 350 |
| Speed | ~350 | 350 |
| Base stat (any) | 255 | 255 |
| SP per stat | 32 | 32 |

*Note: No IVs in Pokémon Champions. Max stat values are lower than in mainline games. Adjust normalisation values once Champions stat calculators publish confirmed maximums for M-A.*

---

## Move Feature Vector (~55 features per move)

Used everywhere a move appears — your moves, opponent revealed moves, opponent prior moves. Same structure throughout. Describes what the move **does**, not which move it **is**.

| Feature | Type | Dim | Notes |
|---|---|---|---|
| Base power normalised | float | 1 | ÷ 250. Status moves = 0 |
| Type one-hot | binary | 18 | 18 types |
| Category | one-hot | 3 | Physical / Special / Status |
| Priority tier | float | 1 | (priority + 7) ÷ 12, range -7 to +5 |
| Accuracy | float | 1 | 0–1. Always-hit = 1.0 |
| PP max normalised | float | 1 | ÷ 40 |
| Is spread (hits both opponents) | binary | 1 | Earthquake, Discharge, Dazzling Gleam etc. |
| Hits ally too | binary | 1 | Earthquake hits partner in doubles |
| Makes contact | binary | 1 | Triggers contact-based abilities |
| Is sound-based | binary | 1 | Bypasses Substitute, triggers Soundproof |
| Is punch-based | binary | 1 | Boosted by Iron Fist |
| Is biting move | binary | 1 | Boosted by Strong Jaw |
| Bypasses protect | binary | 1 | Feint, Phantom Force etc. |
| Is protect variant | binary | 1 | Protect, Wide Guard, Quick Guard, Detect |
| Has recoil | binary | 1 | Double-Edge, Flare Blitz etc. |
| Is healing move | binary | 1 | Recover, Roost, Synthesis etc. |
| Is draining move | binary | 1 | Giga Drain, Leech Life etc. |
| Sets weather | binary | 1 | Rain Dance, Sunny Day, Sandstorm, Snowscape |
| Sets terrain | binary | 1 | Electric/Grassy/Misty/Psychic Terrain |
| Sets Trick Room | binary | 1 | |
| Sets Tailwind | binary | 1 | |
| Is redirection move | binary | 1 | Follow Me, Rage Powder |
| Inflicts status condition | binary | 1 | Thunder Wave, Will-O-Wisp, Spore etc. |
| Status inflicted one-hot | binary | 7 | Burn / Paralysis / Poison / Bad Poison / Freeze / Sleep / None |
| Secondary effect chance | float | 1 | 0–1. 0 if no secondary effect |
| Secondary effect type one-hot | binary | 6 | Flinch / Burn / Paralysis / Freeze / Stat change / None |
| Causes stat change | binary | 1 | |
| Stat change target | one-hot | 3 | Self / Opponent / Both |
| Stat change is boost | binary | 1 | 1 = raises, 0 = lowers |
| Is multi-hit | binary | 1 | Bullet Seed, Icicle Spear etc. |
| Is two-turn move | binary | 1 | Solar Beam, Fly, Dig etc. |
| Is fixed-damage move | binary | 1 | Seismic Toss, Night Shade, OHKO moves |
| **Total** | | **~55** | |

---

## Ability Feature Vector (~25 features per ability)

Describes what the ability **does**, not which ability it **is**. Works for your Pokémon (known) and opponent Pokémon (prior if not yet triggered). New abilities from Mega Evolution are handled by this same feature vector — if a new Mega ability fits existing flags it encodes automatically. If it introduces a genuinely new mechanic, a new flag is added and the encoder is incremented to v3.

| Feature | Type | Dim | Notes |
|---|---|---|---|
| Sets weather on entry | binary | 1 | Drizzle, Drought, Sand Stream, Snow Warning |
| Benefits from specific weather (speed doubled) | binary | 1 | Swift Swim, Chlorophyll, Sand Rush, Slush Rush |
| Benefits from specific weather (other) | binary | 1 | Solar Power, Rain Dish, Sand Force etc. |
| Redirects electric moves | binary | 1 | Lightning Rod |
| Redirects water moves | binary | 1 | Storm Drain |
| Grants type immunity | binary | 1 | Levitate, Flash Fire, Water Absorb, Volt Absorb etc. |
| Immune type one-hot | binary | 18 | Which type is immune. All 0 if no immunity |
| Lowers opponent stat on entry | binary | 1 | Intimidate |
| Has on-contact effect | binary | 1 | Rough Skin, Iron Barbs, Flame Body, Static etc. |
| Prevents own stat drops | binary | 1 | Clear Body, White Smoke, Full Metal Body |
| Sets terrain on entry | binary | 1 | Electric Surge, Grassy Surge, Misty Surge, Psychic Surge |
| Boosts priority of status moves | binary | 1 | Prankster |
| Ignores other abilities | binary | 1 | Mold Breaker, Turboblaze, Teravolt |
| Has speed-related effect | binary | 1 | Speed Boost, Unburden, Quick Feet |
| Enables multi-hit on moves | binary | 1 | Parental Bond (Mega Kangaskhan) |
| Doubles effective Attack | binary | 1 | Pure Power (Mega Medicham), Huge Power |
| Is a Mega-exclusive ability | binary | 1 | Ability only exists post-Mega |
| Is revealed / confirmed in battle | binary | 1 | 1 = triggered and seen. 0 = prior estimate |
| **Total** | | **~25** | |

---

## Mega Evolution Handling

Mega Evolution is a **mid-battle transformation event**. Pre-Mega and post-Mega are encoded as different states. The encoder must handle this explicitly.

**Rules:**
- Only one Mega Evolution allowed per battle (track with a global battle flag)
- Mega Evolution occurs at the start of the turn it is selected, before moves execute
- Mega changes: ability (always), sometimes type, sometimes base stats distribution
- Mega Stone must be held — consumed on Mega Evolution (item slot becomes empty)

**Encoding approach:**
- Pre-Mega: encode base form ability, types, stats
- Post-Mega: encode post-Mega ability, types, stats — loaded from a Mega lookup table
- Add a `has_mega_evolved` binary flag to the active Pokémon block
- Add a `battle_mega_used` binary flag to the battle context block (only one Mega per battle)
- The Mega lookup table is static data: for each eligible Pokémon, store post-Mega ability features, type changes, and stat changes

**Not yet available in M-A (do not include in format filter):**
Mega Sceptile, Mega Blaziken, Mega Swampert, Mega Mawile, Mega Salamence, Mega Metagross, and 17 Legends Z-A Megas (Staraptor, Scolipede, Scrafty, Eelektross, Pyroar, Malamar, Barbaracle, Dragalge, Golisopod, Falinks, Tatsugiri, Baxcalibur, and the three Z-Mega forms of Absol, Garchomp, and Lucario).

---

## Pokémon Feature Block — Your Active Pokémon (×2)

Full information. No uncertainty. Two of these blocks — one per active slot.

### Identity & Typing
| Feature | Type | Dim | Notes |
|---|---|---|---|
| Type 1 one-hot | binary | 18 | Post-Mega type if Mega-evolved |
| Type 2 one-hot | binary | 18 | All 0 if single type. Post-Mega type if changed |
| Is single type | binary | 1 | |
| Has Mega-evolved this battle | binary | 1 | |

### Current Stats (actual values after SP + Stat Alignment, at level 50)
| Feature | Type | Dim | Notes |
|---|---|---|---|
| HP current / HP max | float | 1 | 0–1 ratio |
| HP max normalised | float | 1 | ÷ 400 |
| Attack normalised | float | 1 | ÷ 350. Post-Mega if applicable |
| Defence normalised | float | 1 | ÷ 350 |
| Special Attack normalised | float | 1 | ÷ 350 |
| Special Defence normalised | float | 1 | ÷ 350 |
| Speed normalised | float | 1 | ÷ 350. Reflects Stat Alignment modifier |

### SP Distribution (your team only — fully known)
| Feature | Type | Dim | Notes |
|---|---|---|---|
| SP in HP normalised | float | 1 | ÷ 32 |
| SP in Attack normalised | float | 1 | ÷ 32 |
| SP in Defence normalised | float | 1 | ÷ 32 |
| SP in Special Attack normalised | float | 1 | ÷ 32 |
| SP in Special Defence normalised | float | 1 | ÷ 32 |
| SP in Speed normalised | float | 1 | ÷ 32 |

### Stat Alignment (replaces Nature)
| Feature | Type | Dim | Notes |
|---|---|---|---|
| Boosted stat one-hot | binary | 6 | Which of 6 stats is +10%. All 0 if Serious (neutral) |
| Reduced stat one-hot | binary | 6 | Which of 6 stats is -10%. All 0 if Serious |

### In-Battle Stat Boosts
| Feature | Type | Dim | Notes |
|---|---|---|---|
| Attack boost | float | 1 | (boost + 6) ÷ 12, range -6 to +6 → 0–1 |
| Defence boost | float | 1 | |
| Special Attack boost | float | 1 | |
| Special Defence boost | float | 1 | |
| Speed boost | float | 1 | |
| Accuracy boost | float | 1 | |
| Evasion boost | float | 1 | |

### Primary Status
| Feature | Type | Dim | Notes |
|---|---|---|---|
| Status one-hot | binary | 7 | Healthy / Burn / Paralysis / Poison / Bad Poison / Frozen / Asleep |
| Toxic counter normalised | float | 1 | ÷ 16. 0 if not badly poisoned |
| Turns already asleep normalised | float | 1 | ÷ 3. 0 if not asleep. Model infers wake probability from this |

### Volatile Status (in-battle conditions, reset on switch)
| Feature | Type | Dim | Notes |
|---|---|---|---|
| Confused | binary | 1 | |
| Taunted | binary | 1 | Cannot use status moves |
| Taunt turns remaining normalised | float | 1 | ÷ 4 |
| Encored | binary | 1 | Locked into one move |
| Encore turns remaining normalised | float | 1 | ÷ 3 |
| Encore move slot index normalised | float | 1 | 0–3 ÷ 3. Which move slot is locked |
| Leech seeded | binary | 1 | |
| Substitute active | binary | 1 | |
| Partially trapped | binary | 1 | Fire Spin, Wrap etc. |
| Trap turns remaining normalised | float | 1 | ÷ 5 |
| Infatuated | binary | 1 | |
| Disabled | binary | 1 | A specific move is disabled |
| Disabled move slot normalised | float | 1 | 0–3 ÷ 3 |
| Powered up / charging | binary | 1 | Two-turn move charge turn, Power Herb used |
| Identified | binary | 1 | Odor Sleuth / Miracle Eye active |

### Ability (post-Mega if applicable)
| Feature | Type | Dim | Notes |
|---|---|---|---|
| Ability feature vector | float | 25 | See Ability Feature Vector. Post-Mega ability if Mega-evolved |
| Ability has activated this battle | binary | 1 | Whether it has triggered at least once |

### Item
| Feature | Type | Dim | Notes |
|---|---|---|---|
| Has item | binary | 1 | 0 after Mega Stone consumed, Knock Off etc. |
| Is Mega Stone | binary | 1 | Consumed on Mega Evolution |
| Is choice item | binary | 1 | Choice Band / Specs / Scarf |
| Is berry | binary | 1 | |
| Berry triggers on HP threshold | binary | 1 | Sitrus, Pinch berries |
| Berry threshold normalised | float | 1 | HP % at which berry triggers |
| Is Life Orb | binary | 1 | |
| Is Assault Vest | binary | 1 | Prevents status move selection |
| Is Rocky Helmet | binary | 1 | |
| Is Eject Button / Eject Pack | binary | 1 | |
| Boosts a specific stat | binary | 1 | Muscle Band, Wise Glasses, type-boosting items |
| Provides speed boost | binary | 1 | Choice Scarf |
| Item type one-hot | one-hot | 8 | Choice / Berry / Orb / Mega Stone / Seed / Held-battle / Held-passive / None |

### Choice Lock
| Feature | Type | Dim | Notes |
|---|---|---|---|
| Is choice locked | binary | 1 | Used a move while holding a Choice item |
| Locked move slot normalised | float | 1 | 0–3 ÷ 3. Which slot is locked. 0 if not locked |

### Moves (×4 slots)
| Feature | Type | Dim | Notes |
|---|---|---|---|
| Move feature vector | float | 55 | See Move Feature Vector |
| PP remaining normalised | float | 1 | Current PP ÷ max PP |
| Is selectable this turn | binary | 1 | 0 if: no PP / taunted + status move / encored + wrong move / disabled / choice locked + wrong move / Assault Vest + status move |

*4 move slots × (55 + 2) = 228 dim*

### Position & Mega Status
| Feature | Type | Dim | Notes |
|---|---|---|---|
| Field position | binary | 1 | 0 = left slot, 1 = right slot |
| Has Mega Stone equipped (pre-Mega) | binary | 1 | Can still Mega if battle Mega not yet used |

**Your Active Pokémon block total: ~340 dim per Pokémon × 2 = ~680 dim**

---

## Pokémon Feature Block — Your Bench Pokémon (×2)

Full information. No volatile status (not active). Retain full SP, Stat Alignment, and move knowledge.

| Feature | Type | Dim | Notes |
|---|---|---|---|
| Slot is occupied | binary | 1 | 0 if fainted |
| Type 1 + Type 2 one-hot | binary | 36 | |
| Is single type | binary | 1 | |
| HP current / HP max | float | 1 | |
| HP max normalised | float | 1 | ÷ 400 |
| All 5 other actual stats normalised | float | 5 | Atk/Def/SpA/SpD/Spe ÷ 350 each |
| SP distribution normalised × 6 | float | 6 | ÷ 32 per stat |
| Boosted stat one-hot (Stat Alignment) | binary | 6 | |
| Reduced stat one-hot | binary | 6 | |
| Status one-hot | binary | 7 | |
| Toxic counter normalised | float | 1 | |
| Turns already asleep normalised | float | 1 | |
| Ability feature vector | float | 25 | Full ability encoding — known for bench |
| Has Mega Stone | binary | 1 | And hasn't yet Mega-evolved |
| Can Mega-evolve (if battle Mega not yet used) | binary | 1 | |
| Has item | binary | 1 | |
| Is choice item | binary | 1 | |
| Item type one-hot | one-hot | 8 | |
| Moves × 4 (full feature vector) | float | 220 | 55 dim per move |
| PP remaining normalised × 4 | float | 4 | |
| Is selectable if switched in × 4 | binary | 4 | Taunt, no PP etc. |

**Your Bench block total: ~340 dim per Pokémon × 2 = ~680 dim**

---

## Pokémon Feature Block — Opponent Active Pokémon (×2)

Partial information. Observable values encoded directly. Unknown values use feature-level priors from Smogon usage stats / Champions ladder data, with explicit revealed flags.

### Identity & Observable State
| Feature | Type | Dim | Notes |
|---|---|---|---|
| Type 1 + Type 2 one-hot | binary | 36 | Known from OTS. Post-Mega type if Mega-evolved |
| Is single type | binary | 1 | |
| Has Mega-evolved this battle | binary | 1 | Visible event |
| HP current % | float | 1 | Visible in battle |
| Base stats normalised × 6 | float | 6 | Known from species ÷ 255 each |
| Estimated actual Speed normalised | float | 1 | Base speed + common SP spread estimate |

### Observable In-Battle State
| Feature | Type | Dim | Notes |
|---|---|---|---|
| All 7 stat boosts | float | 7 | Visible |
| Status one-hot | binary | 7 | Visible |
| Toxic counter normalised | float | 1 | |
| Turns already asleep normalised | float | 1 | |
| All volatile status fields | float | 15 | Same set as your active — all visible |

### Ability (post-Mega if applicable)
| Feature | Type | Dim | Notes |
|---|---|---|---|
| Ability feature vector | float | 25 | If triggered: actual post-Mega ability encoded. If not: weighted prior from Champions usage data |
| Ability is confirmed | binary | 1 | 1 = triggered and observed. 0 = prior estimate |

### Item
| Feature | Type | Dim | Notes |
|---|---|---|---|
| Item is confirmed | binary | 1 | Revealed in battle (Knocked Off, consumed berry, Life Orb recoil observed) |
| Is Mega Stone (confirmed or prior) | binary | 1 | |
| Mega Stone consumed (post-Mega) | binary | 1 | |
| Is choice item (confirmed or inferred) | binary | 1 | Observed using same move twice |
| Is Life Orb (confirmed or prior) | binary | 1 | Recoil pattern observed |
| Item type one-hot | one-hot | 8 | From confirmed or prior |
| Item confidence | float | 1 | 0 = pure prior, 1 = confirmed |

### Choice Lock (inferred)
| Feature | Type | Dim | Notes |
|---|---|---|---|
| Inferred choice locked | binary | 1 | Same move used twice + choice item prior |
| Locked move slot normalised | float | 1 | Which slot, 0 if unknown |
| Lock is confirmed | binary | 1 | |

### Moves (×4 slots)
| Feature | Type | Dim | Notes |
|---|---|---|---|
| Move feature vector | float | 55 | If revealed: actual move. If not: weighted average of top common moves for this species from Champions ladder data |
| Move is revealed | binary | 1 | 1 = seen in battle. 0 = prior estimate |
| PP pressure estimate normalised | float | 1 | Times used ÷ max PP. Proxy for PP depletion |

*4 × (55 + 2) = 228 dim*

**Opponent Active block total: ~310 dim per Pokémon × 2 = ~620 dim**

---

## Pokémon Feature Block — Opponent Bench

This is the most complex block because of the 6-pick-4 format. The opponent registered 6 Pokémon, chose 4 to bring, and leads with 2. Their bench contains exactly **2 Pokémon** — but you don't know which 2 of the remaining 4 from their roster they brought. The other 2 stayed home.

### Presence States

Each non-active Pokémon from the opponent's roster of 6 is in one of four states at any point in the battle:

| State | Meaning | When |
|---|---|---|
| **Confirmed benched** | Seen in battle, currently inactive | Was active previously, switched or forced out |
| **Confirmed fainted** | Was in battle, now fainted | Eliminated |
| **Possibly brought** | Not yet seen — may be one of their 2 bench slots | Unknown until revealed |
| **Confirmed absent** | Definitely left home | Enough other Pokémon seen to rule this out |

**Probability of being in battle (P_present):**

As the battle progresses you can compute this precisely:

```
P_present = unseen_bench_slots_remaining ÷ unseen_pokemon_in_roster_remaining
```

Examples:
- Battle start, 2 leads revealed: 4 unseen Pokémon, 2 bench slots → P = 0.50 each
- After seeing 3rd Pokémon: 3 unseen, 1 bench slot → P = 0.33 each
- After seeing 4th Pokémon: 2 unseen, 0 bench slots → P = 0.0 → confirmed absent

### Opponent Bench Block Structure

4 slots covering all possible non-active roster Pokémon. Always 4 slots regardless of battle state — unused slots are zero-padded with presence state = Confirmed Absent.

Per slot:

| Feature | Type | Dim | Notes |
|---|---|---|---|
| Presence state one-hot | binary | 4 | Confirmed benched / Confirmed fainted / Possibly brought / Confirmed absent |
| P(in this battle) | float | 1 | 0.0–1.0. Updated each turn as Pokémon are revealed |
| Type 1 + Type 2 one-hot | binary | 36 | Known from OTS |
| Is single type | binary | 1 | |
| Base stats normalised × 6 | float | 6 | Known from species |
| HP % last known | float | 1 | If ever been active. -1 flag if never active |
| HP is known | binary | 1 | 1 if HP observed this battle |
| Status one-hot | binary | 7 | If known from being active. 0 if never active |
| Status is known | binary | 1 | |
| Ability feature vector | float | 25 | Prior if not revealed. Actual if triggered |
| Ability is confirmed | binary | 1 | |
| Has Mega Stone (confirmed or prior) | binary | 1 | |
| Has Mega-evolved | binary | 1 | If has been active and Mega-evolved |
| Item type one-hot | one-hot | 8 | Prior if not revealed |
| Item is confirmed | binary | 1 | |
| Moves × 4 (feature vector) | float | 220 | Prior if not revealed. Actual for revealed moves |
| Move revealed flag × 4 | binary | 4 | |
| Has fainted | binary | 1 | |

*~325 dim per slot × 4 slots = ~1300 dim*

**Opponent Bench block total: ~1300 dim**

---

## Field State

Fully observable. No uncertainty. No Terastallization in this format.

### Weather
| Feature | Type | Dim | Notes |
|---|---|---|---|
| Weather one-hot | binary | 5 | None / Rain / Sun / Sand / Snow |
| Turns remaining normalised | float | 1 | ÷ 8. 1.0 = permanent (ability-based) |
| Is permanent | binary | 1 | Set by ability, not by move |
| Set by your side | binary | 1 | |

### Terrain
| Feature | Type | Dim | Notes |
|---|---|---|---|
| Terrain one-hot | binary | 5 | None / Electric / Grassy / Misty / Psychic |
| Turns remaining normalised | float | 1 | ÷ 8 |

### Trick Room
| Feature | Type | Dim | Notes |
|---|---|---|---|
| Active | binary | 1 | |
| Turns remaining normalised | float | 1 | ÷ 5 |

### Tailwind
| Feature | Type | Dim | Notes |
|---|---|---|---|
| Your side active | binary | 1 | |
| Your turns remaining normalised | float | 1 | ÷ 4 |
| Opponent side active | binary | 1 | |
| Opponent turns remaining normalised | float | 1 | ÷ 4 |

### Screens
| Feature | Type | Dim | Notes |
|---|---|---|---|
| Your Light Screen active + turns | binary + float | 2 | |
| Your Reflect active + turns | binary + float | 2 | |
| Opponent Light Screen active + turns | binary + float | 2 | |
| Opponent Reflect active + turns | binary + float | 2 | |

### Other Field Effects
| Feature | Type | Dim | Notes |
|---|---|---|---|
| Gravity active + turns remaining | binary + float | 2 | ÷ 5 |
| Your Safeguard active + turns | binary + float | 2 | |
| Opponent Safeguard active + turns | binary + float | 2 | |
| Magic Room active + turns | binary + float | 2 | Items suppressed |
| Wonder Room active + turns | binary + float | 2 | Def and SpDef swapped |

**Field state total: ~40 dim**

---

## Battle Context

### Momentum & Score
| Feature | Type | Dim | Notes |
|---|---|---|---|
| Turn number normalised | float | 1 | ÷ 30, capped at 1.0 |
| Your Pokémon remaining normalised | float | 1 | ÷ 4 |
| Opponent Pokémon remaining normalised | float | 1 | ÷ 4 |
| Your total HP % across all remaining | float | 1 | Average HP% across your surviving Pokémon |
| Opponent total HP % across all remaining | float | 1 | |
| KO differential normalised | float | 1 | (your KOs − their KOs + 4) ÷ 8 → 0–1 |
| Battle Mega used (your side) | binary | 1 | Only one Mega per battle allowed |
| Battle Mega used (opponent side) | binary | 1 | |

### Last Turn Actions
| Feature | Type | Dim | Notes |
|---|---|---|---|
| Your P1 used protect last turn | binary | 1 | Critical for protect cycle prediction |
| Your P2 used protect last turn | binary | 1 | |
| Opponent P1 used protect last turn | binary | 1 | |
| Opponent P2 used protect last turn | binary | 1 | |
| Your P1 last move type one-hot | binary | 18 | Type of move used. All 0 if switched or turn 1 |
| Your P2 last move type one-hot | binary | 18 | |
| Opponent P1 last move type one-hot | binary | 18 | If observed |
| Opponent P2 last move type one-hot | binary | 18 | |
| Your P1 switched last turn | binary | 1 | |
| Your P2 switched last turn | binary | 1 | |
| Opponent P1 switched last turn | binary | 1 | |
| Opponent P2 switched last turn | binary | 1 | |
| Your P1 used Mega last turn | binary | 1 | Mega occurred this battle on your side |
| Opponent P1/P2 used Mega last turn | binary | 1 | |

### Damage Last Turn
| Feature | Type | Dim | Notes |
|---|---|---|---|
| Damage your P1 took last turn % | float | 1 | As % of max HP |
| Damage your P2 took last turn % | float | 1 | |
| Damage opponent P1 took last turn % | float | 1 | |
| Damage opponent P2 took last turn % | float | 1 | |

### Revealed Information Tracker
| Feature | Type | Dim | Notes |
|---|---|---|---|
| Has each opponent Pokémon been active × 6 | binary | 6 | One per roster slot |
| Moves revealed count per opponent Pokémon × 6 | float | 6 | ÷ 4 |
| Item confirmed per opponent Pokémon × 6 | binary | 6 | |
| Ability confirmed per opponent Pokémon × 6 | binary | 6 | |

**Battle context total: ~130 dim**

---

## Full Encoding Summary

| Block | Content | Approx Dim |
|---|---|---|
| Your active Pokémon × 2 | Full info, SP, Stat Alignment, Mega state, moves, ability, item, volatiles | ~680 |
| Your bench Pokémon × 2 | Full info, no volatile status | ~680 |
| Opponent active Pokémon × 2 | Partial info, priors for unknowns, revealed flags, Mega state | ~620 |
| Opponent bench × 4 slots | Presence states, P(in battle), partial info per slot | ~1300 |
| Field state | Weather, terrain, TR, Tailwind, screens, other effects | ~40 |
| Battle context | Turn, KOs, HP differential, Mega tracking, last turn actions | ~130 |
| **Total** | | **~3450 dim** |

---

## Action Space

The model outputs a **joint action** for both active Pokémon simultaneously — not independent decisions per Pokémon.

### Per Pokémon Action Options

| Action | Conditions |
|---|---|
| Use move 1/2/3/4 → target opponent left | Move selectable, single-target |
| Use move 1/2/3/4 → target opponent right | Move selectable, single-target |
| Use move 1/2/3/4 → target self | Move selectable, self-targeting |
| Use move 1/2/3/4 → target partner | Move selectable, partner-targeting |
| Use move 1/2/3/4 (spread, no target) | Spread move — hits both opponents automatically |
| Use move + Mega Evolution | If Mega Stone held and battle Mega not yet used |
| Switch to bench slot 1 | Bench slot 1 occupied and not fainted |
| Switch to bench slot 2 | Bench slot 2 occupied and not fainted |

Roughly **~12–16 options per Pokémon** depending on move set, bench state, and Mega availability.

### Joint Action Space
~14 × ~14 = **~196 joint action combinations** (upper bound — most battles have fewer valid options)

### Action Masking
Invalid joint actions masked to −∞ before softmax output:
- Both Pokémon cannot switch to the same bench slot
- Cannot select a move with 0 PP
- Cannot use status moves if taunted
- Cannot use a non-choice-locked move if choice locked
- Cannot use a non-encored move if encored
- Cannot use a disabled move
- Cannot use status moves with Assault Vest equipped
- Cannot select Mega if battle Mega already used this battle
- Cannot select Mega if no Mega Stone held

---

## Encoding Versioning

This is **EncoderV2**. Interface between encoder and model is fixed:

```
GameState  →  EncoderV2  →  Tensor (~3450 dim)  →  Projection layer  →  Policy network
```

Changes between versions are documented in `ENCODING_CHANGELOG.md`. Incrementing to v3 requires retraining the projection layer but not the full policy network, provided the projected output dimension stays the same.

**Triggers for EncoderV3:**
- New format introduces Terastallization or Z-Moves
- New Mega ability introduces a mechanic not covered by existing ability flags
- New item type introduced with substantially different mechanics
- Regulation M-B or later adds Pokémon with novel mechanics not encodable in existing feature vectors

---

## Prior Data Sources

For opponent encoding (unknown moves, items, abilities), priors come from:

- **Pokémon Champions ladder usage stats** — primary source once the format matures (2–4 weeks after launch)
- **Early meta analysis** (Smogon VGC forums, Victory Road) — used for the first weeks before ladder data is available
- **Limitless tournament data** — high quality but low volume; available from Indianapolis Regionals onwards (May 29–31)

Update priors monthly as the meta evolves. Prior staleness is a known risk — a set that was 70% common in week 1 may drop to 20% by week 6 as the meta develops.

---

*Stage 3: State Encoding v2 | Pokémon Champions Regulation M-A | VGC AI System*
