# Stage 1 & 2: Team Selection + Lead Selection — Implementation Brief
**Pokémon VGC AI System — Standalone Module**

---

## Purpose

This brief covers the complete implementation of the Team & Lead Selection module as a standalone, independently deployable component. It can be built, tested, and validated in isolation from all other stages of the VGC AI system.

This module combines what were originally Stage 1 (team selection) and Stage 2 (lead selection) into a single jointly optimised model. The two decisions are made simultaneously rather than sequentially because they are not truly independent — the best 4 to bring is partly determined by which lead pairs they enable, and the best lead pair depends on which 4 were brought.

---

## Context & Format Rules

This module operates under **Open Team Sheet (OTS)** rules — the current VGC standard.

**What both players know before any decision is made:**
- Their own full roster of 6 with complete sets (species, moves, item, EVs, nature, ability)
- The opponent's full roster of 6 — species only, no sets

**What neither player knows:**
- Which 4 the opponent brought
- Which 2 the opponent is leading

This is a **symmetric information problem**. Both players make their bring and lead decisions with identical quality of information — full knowledge of their own side, species-only knowledge of the opponent's full 6, and no knowledge of which 4 the opponent selected or intends to lead.

---

## What the Module Does

```
INPUT:
  Your roster  — 6 Pokémon with full sets (species, moves, item, EVs, nature, ability)
  Opponent     — 6 Pokémon species only

OUTPUT:
  Which 4 of your 6 to bring  (bring decision)
  Which 2 of those 4 to lead  (lead decision)
  Expressed as a single ranked list of (bring-4, lead-2) pairs

HOW:
  A trained model scores all valid (bring-4, lead-2) combinations
  from your roster against the expected opponent configuration.
  The top-ranked combination is selected.
```

The model is not outputting a win probability. It is outputting a **joint ranking** of bring+lead combinations — which complete pre-battle configuration gives you the best position in expectation across the plausible opponent configurations they might bring and lead.

---

## Infrastructure Assumptions

- **Showdown server** running in a Docker container — accessible via WebSocket for battle simulation
- **Python environment** in a separate Docker container — all pip packages available, communicates with Showdown container over the Docker network
- No dependency on any other stage of the VGC AI system
- No PPO agent required — a heuristic battle agent is built as part of this module

---

## The Joint Decision Space

### Team Selection alone
C(6,4) = **15 possible combinations** of 4 from your roster of 6

### Lead Selection alone
C(4,2) = **6 possible lead pairs** from your chosen 4

### Combined
15 × 6 = **90 possible (bring-4, lead-2) combinations** from your roster of 6

This is still a tractable space. The model evaluates all 90 and selects the best one. During training, the experiment runner exhaustively tests all 90 combinations across diverse matchups to build the ground truth dataset.

---

## Why Joint Rather Than Sequential

**The decisions are not independent.** A Pokémon that scores mediocrely on team selection alone may be the key to your best lead pair. A sequential model — pick 4, then pick 2 — would leave it home. A joint model recognises its lead value and brings it.

**The input is identical for both decisions.** Both are made with the same information (your 6 full sets, opponent 6 species only) before any new information arrives. There is no natural boundary where a separate model adds value from new data.

**Joint training finds emergent synergies.** Certain combinations of 4 are only strong because of the lead pair they enable. Sequential models miss this. Joint training discovers it naturally from win rate data.

**Simpler system overall.** One model, one training loop, one dataset, one interface. Less to build, less to debug, less to maintain.

---

## Information Asymmetry

| | Your Team | Opponent Team |
|---|---|---|
| Species | ✅ Known | ✅ Known |
| Types | ✅ Known | ✅ Known |
| Base Stats | ✅ Known | ✅ Known |
| Moves | ✅ Known | ❌ Use Smogon usage priors |
| Item | ✅ Known | ❌ Use Smogon usage priors |
| EVs / Nature | ✅ Known | ❌ Use Smogon usage priors |
| Actual Final Stats | ✅ Calculable | ❌ Estimated only |
| Which 4 they brought | ❌ Unknown | — |
| Which 2 they lead | ❌ Unknown | — |

The model must reason about the **expected opponent configuration** — a probability-weighted average over the plausible (bring-4, lead-2) combinations the opponent might play. This is the core challenge of the lead component: you are optimising against a distribution of opponents, not a single known opponent.

---

## Opponent Modelling

Because you don't know their 4 or their lead, the model must reason about what they are likely to do. This is handled implicitly by training on diverse matchups — the model learns which of your configurations performs well in expectation across many possible opponent strategies.

There is also an explicit opponent modelling component built into the encoding:

**Smogon usage priors** fill in the unknown opponent sets — ability, item, and move probabilities per species at high level. This gives the model a principled estimate of the opponent's likely sets even without knowing them.

**Role tags** inferred from species + common sets give the model a sense of the opponent's likely strategic role (setter, abuser, redirector, speed control etc.) which is more actionable than raw stat priors.

The symmetry of the problem is also useful: the model learns what your opponent would reasonably bring against your 6, because it learns what you would bring against various opponents. The opponent is running the same logic in reverse.

---

## Encoding

Each Pokémon is converted to a fixed-size numeric feature vector. The encoding is the most important design decision — it determines whether the model **generalises** to unseen teams or **memorises** training teams.

**Key principle:** encode features, not identities. A model that learns "bring Pokémon #456 when opponent has Pokémon #123" memorises. A model that learns "bring your redirector when opponent has a fast spread attacker" generalises.

### Your Team (~183 features per Pokémon)

| Feature Group | Details | Dim |
|---|---|---|
| Species embedding | Learned 32-dim vector per species | 32 |
| Types | One-hot type1 + type2 | 36 |
| Base stats | HP/Atk/Def/SpA/SpD/Spe normalised ÷ 255 | 6 |
| EVs | Each stat EV normalised ÷ 252 | 6 |
| Actual stats | Computed from base + EV + nature, normalised ÷ 500 | 6 |
| Nature | One-hot across 25 natures | 25 |
| Ability | Learned embedding | 16 |
| Item | Learned embedding | 16 |
| Moves × 4 | Learned embedding per move | 64 |
| Role tags | Binary flags (see below) | 10 |

**Role tags — computed heuristics, not learned:**
- Is Trick Room setter (has Trick Room in moveset)
- Is Trick Room abuser (base Spe ≤ 60)
- Is weather setter (has Rain Dance, Drought, Sandstorm, Snow or relevant ability)
- Is terrain setter (has Electric/Grassy/Misty/Psychic Terrain)
- Is redirector (has Follow Me or Rage Powder)
- Is spread attacker (has Earthquake, Discharge, Surf, Dazzling Gleam etc.)
- Has Protect / Wide Guard / Quick Guard
- Is bulky (base HP + Def + SpD > 220)
- Is fast (base Spe > 100)
- Is speed control (has Tailwind, Icy Wind, Electroweb, Thunder Wave etc.)

### Opponent Team (~148 features per Pokémon)

| Feature Group | Details | Dim |
|---|---|---|
| Species embedding | Shared learned embedding with your encoding | 32 |
| Types | One-hot type1 + type2 | 36 |
| Base stats | Normalised ÷ 255 | 6 |
| Ability prior | Probability distribution from Smogon usage stats | 16 |
| Item prior | Probability distribution from Smogon usage stats | 16 |
| Move priors × 4 | Top 4 most common moves weighted by usage % | 64 |
| Role tags | Inferred from species + common sets | 10 |

Species embeddings are **shared** between your encoding and the opponent encoding. Both sides use the same learned embedding table — this ensures the model builds a unified representation of what each species means regardless of which side it appears on.

---

## Model Architecture

### Phase 1: MLP Baseline

Build this first. Gets the data pipeline and training loop working before investing in a more complex architecture.

```
Flatten opponent 6  →  MLP encoder  →  128-dim opponent context

For each of the 90 (bring-4, lead-2) combinations:
  Encode your 4 brought Pokémon
  Encode your 2 lead Pokémon (with lead flag)
  Concatenate with opponent context
  MLP scorer → scalar score

Select combination with highest score
```

The lead Pokémon get an additional **lead flag** (binary 1/0) appended to their encoding so the model knows which 2 of the 4 are leading. The back 2 get a 0 flag. This lets the model learn lead-specific patterns without a separate architecture.

### Phase 2: Transformer with Cross-Attention (Target Architecture)

```
Your 6    →  your_proj  (Linear 183→128)  →  (6, 128)  ─┐
                                                           ├─ cross-attention
Opponent 6 →  opp_proj  (Linear 148→128)  →  (6, 128)  ─┘
                  ↓
      Self-attention over your 6
                  ↓
      Each of your 6 attends to opponent 6 (cross-attention)
                  ↓
      Pokémon-level embeddings: (6, 128)
                  ↓
      For each (bring-4, lead-2) combination:
        Pool the 4 brought embeddings
        Pool the 2 lead embeddings separately (with lead-specific MLP)
        Concatenate: [bring_pool | lead_pool]  →  combination score
                  ↓
      Rank all 90 combinations by score
      Return top-ranked (bring-4, lead-2)
```

Two separate projection layers map each side into the shared embedding space. The bring and lead components have separate pooling MLPs so the model can learn bring-specific and lead-specific patterns independently before combining them into a joint score.

### Why Not Use the PPO Agent for Training?

The existing PPO agent is Trick Room-specialised. Using it to generate training battles would bias the signal — Trick Room configurations would appear to win more often because the agent plays them better, not because they are structurally superior. The heuristic agent has no stylistic bias and produces clean, unconfounded signal.

---

## Heuristic Battle Agent

A simple rule-based agent that plays battles on the Showdown server. Used exclusively for generating training data — not used in production.

**Decision rules (in priority order):**
1. If a move KOs an active opponent and is super effective — use it, target the frailer opponent
2. If HP < 30% and Protect is available — use Protect
3. Use highest base power move available, target the opponent with lower bulk
4. If a teammate is at risk of being KO'd and you have a super effective move on the attacker — prioritise that target
5. Protect if no good attacking option
6. Random legal move as fallback

**What the heuristic agent does NOT do:**
- Set up Trick Room or weather strategically
- Switch Pokémon
- Use status moves
- Make reads on opponent behaviour

This is intentional. The heuristic agent needs to be **consistent and unbiased**, not strong. Win rate differences between configurations reflect structural quality (type coverage, bulk, role balance) not play quality. The lead pair that gets Trick Room up on turn 1 will show a measurable win rate advantage even with a dumb in-battle agent — that's the signal you want.

---

## Training Data Generation — The Experiment Runner

### The 10 Tournament Teams

Source 10 real VGC tournament teams with **full sets** (species, moves, item, EVs, nature, ability). Aim for diverse archetypes across the pool:

- 2 × Trick Room teams
- 2 × Rain teams
- 2 × Hyper Offence teams
- 2 × Sun or Sand teams
- 2 × Bulky Control / Balance teams

**Sources:** Limitless (limitlessvgc.com) publishes full team sheets for Regional and above events. Smogon VGC forums also publish tournament teams. Use Gen 9 VGC 2025 format teams only.

### Experiment Matrix

```
For each team A (10 options — as your roster):
  For each team B ≠ A (9 options — as opponent):          # 90 roster pairings
    For each (bring-4, lead-2) combination from A:         # up to 90 combinations
      Battle this configuration vs B's heuristic selection, N=20 times
      Record: wins, losses, win_rate, avg_turn1_hp_delta, avg_win_margin
```

**Opponent's selection during training:** The opponent uses a heuristic selection — highest base stat total from their 6 as the bring-4, and the fastest + most offensive pair as the lead-2. This is a simplification that avoids needing an opponent model during the bootstrap phase. Flag for Phase 2.

**Volume estimate:**
- 90 roster pairings × 90 combinations × 20 battles = **162,000 battles**
- On a local Showdown server this takes roughly 12–24 hours
- Start with 10 sampled combinations per pairing (18,000 battles) to validate the pipeline

**Sampling strategy:** Don't run all 90 combinations immediately. Sample 10 random (bring-4, lead-2) combinations per pairing first. Once you have baseline win rates, identify pairings where top combinations are close (win rate difference < 10%) and run those to exhaustion. Pairings with clear winners (one combination dominates) need fewer samples.

### Output Dataset

Each record:

```json
{
  "your_roster":       "6 Pokémon with full sets",
  "opp_roster":        "6 Pokémon species only",
  "bring_indices":     "[4 indices into your_roster]",
  "lead_indices":      "[2 indices into bring_indices]",
  "wins":              "int",
  "losses":            "int",
  "win_rate":          "float  — primary training signal",
  "avg_turn1_hp_delta": "float — your total HP minus opponent total HP after turn 1",
  "avg_win_margin":    "float — avg Pokémon remaining at end (positive = won by more)"
}
```

**Turn 1 HP delta** is the key new metric vs the team-selection-only brief. It directly measures lead quality — a good lead pair takes less damage and deals more on turn 1 regardless of whether the overall battle is won or lost. This gives the lead component a cleaner training signal than full battle win rate alone.

### Training Labels

For each (your_roster, opp_roster) pairing, each (bring-4, lead-2) combination gets two scores:

```
bring_score[combination] = avg win_rate across all lead pairs within this bring-4
lead_score[combination]  = win_rate of this specific (bring-4, lead-2)
                         + 0.3 × avg_turn1_hp_delta (shaped reward for lead quality)
```

The model trains on **lead_score** as the primary target — this captures both team quality and lead quality jointly. The bring_score is used as an auxiliary loss to ensure the team selection component learns independently of lead noise.

---

## Training Loop

### Phase 1: Train on Experiment Data

```
Dataset:   (your_roster_encoded, opp_roster_encoded) → 90 combination scores
Loss:      MSE on lead_score (primary) + MSE on bring_score (auxiliary, 0.3 weight)
Metric:    Does top-ranked combination match or beat the best known from exhaustive search?
```

### Phase 2: Expand and Retrain

Add more tournament teams (target 50+), regenerate experiment data, retrain. More teams = better generalisation. Priority is diversity of archetypes, not raw volume.

### Phase 3: RL Fine-Tuning (Later — Requires General PPO Agent)

Once the PPO agent has been generalised beyond Trick Room, fine-tune the model end-to-end using actual battle outcomes. The team + lead selection module receives a reward signal based on whether the chosen configuration won and by what margin. This phase is out of scope for the initial milestone.

---

## Validation & Success Criteria

The module is complete when it passes all four checks:

### 1. Held-Out Generalisation
Hold 2 of the 10 teams out of training entirely. Test the model on matchups involving those 2 unseen rosters.

**Target:** Top-ranked (bring-4, lead-2) combination matches or beats the best known combination from exhaustive search in ≥ 55% of held-out matchups

### 2. Team Selection Sanity Checks

| Scenario | Expected behaviour |
|---|---|
| Opponent has 4 fast Pokémon, you have a TR setter | Leave TR setter home |
| Opponent has a redirector, you have spread attackers | Bring single-target nukes |
| Opponent is all physical attackers, you have a physical wall | Bring the wall |
| Your 4 best coverage Pokémon clearly dominate | Select them regardless of opponent |

### 3. Lead Selection Sanity Checks

| Scenario | Expected behaviour |
|---|---|
| You brought a TR setter + abuser | Lead with setter + abuser |
| You brought Tailwind setter + sweeper | Lead setter + sweeper |
| Opponent's 6 are all fast, you have redirection | Lead redirector + your fastest attacker |
| Your lead pair has a type disadvantage vs likely opponent leads | Prefer alternate lead pair |

### 4. Win Rate vs Random
Model's recommended (bring-4, lead-2) vs a randomly selected combination from the same roster, N=100 battles each.

**Target:** Model selection wins ≥ 65% of games vs random combination from same roster

---

## Data Requirements Summary

| Data | Source | Used For |
|---|---|---|
| 10 VGC tournament teams (full sets) | Limitless / Smogon VGC forums | Training + opponent pool |
| PokéAPI — base stats, types, learnsets | api.pokemonapi.co / local cache | Pokémon encoder |
| Smogon usage stats (Gen 9 VGC 2025) | smogon.com/stats | Opponent ability/item/move priors |
| Experiment runner output | Generated locally (~162k battles) | Training labels |
| Held-out matchup results | Generated locally | Validation |

---

## Tech Stack

| Component | Technology | Purpose |
|---|---|---|
| Battle simulation | Pokémon Showdown (Docker) | Running heuristic battles |
| Showdown client | `websockets` Python library | Connecting Python to Showdown container |
| Pokémon encoder | PyTorch `nn.Embedding` + numpy | Feature vectors for both sides |
| Heuristic agent | Pure Python | Driving battles during data generation |
| Experiment runner | Python async + Showdown client | Cycling pairings, combinations, storing results |
| Dataset storage | JSON + Pandas | Storing and loading experiment results |
| Model — Phase 1 | PyTorch MLP | Baseline joint model |
| Model — Phase 2 | PyTorch Transformer + cross-attention | Target joint model |
| Training loop | PyTorch + Adam optimiser | Model training with dual loss |
| Usage stats fetcher | `requests` + JSON parser | Smogon usage data for opponent priors |
| Experiment tracking | Weights & Biases or MLflow | Tracking training runs and metrics |

---

## Key Watchouts

**Encoding drives generalisation — get this right first**
Raw species ID encoding memorises training teams. Rich feature vector encoding (types, stats, roles) generalises. This is non-negotiable. Validate that similar Pokémon produce similar embeddings before training the model.

**Lead flag is essential**
Without the lead flag the model cannot distinguish which 2 of the 4 are leading. Every Pokémon in the bring-4 looks identical to the model. The lead flag is a simple binary feature appended to each Pokémon's encoding — 1 if leading, 0 if in back. Do not skip this.

**Turn 1 HP delta noise**
The heuristic agent doesn't play turn 1 well (no strategic switching, no reads). Turn 1 HP delta is still a useful signal for structural lead quality but will be noisier than full battle win rate. Weight it at 0.3 in the combined label, not higher.

**90 combinations per pairing is expensive**
162,000 battles is significant compute. Start with 10 sampled combinations per pairing (18,000 battles) to validate everything end to end before committing to the full run. Use checkpointing — save results after every pairing so a crash doesn't lose hours of data.

**Role overlap in bring selection**
Two high-scoring Pokémon may fill the same role (e.g. two TR setters). Add a post-processing synergy check: if the top combination contains role duplicates, consider the next best non-overlapping combination before committing.

**Opponent heuristic selection is a simplification**
The opponent brings their top-stat-total 4 and leads their fastest 2 during training. Real opponents make strategic decisions. This means the model may underperform against opponents who specifically counter your team's win condition. Addressed in Phase 2 when opponent selection becomes model-driven.

**Showdown container stability**
Running 162,000 battles requires the Showdown container to be stable for many hours. Build retry logic with exponential backoff, battle timeouts, and checkpoint saves after every completed pairing.

**Data freshness**
Tournament teams and Smogon usage stats reflect the meta at collection time. Track source dates. Weight teams from the last 2 months of the current format more heavily. Re-pull Smogon usage stats monthly.

**Small dataset generalisation**
Even with 10 teams × 90 combinations = 900 training records per epoch, this is a small dataset. Use dropout (0.2–0.3), weight decay (1e-4), and early stopping on held-out loss. The primary lever for better generalisation is more teams, not more epochs.

---

## Build Order

```
Step 1 — Data sourcing
  Source 10 tournament teams with full sets (diverse archetypes)
  Pull Smogon usage stats for Gen 9 VGC 2025
  Pull base stats, types, learnsets from PokéAPI
  Cache all data locally in JSON

Step 2 — Pokémon encoder
  Implement full encoding for your team (~183 features)
  Implement prior-based encoding for opponent team (~148 features)
  Implement lead flag appended to your encoding (184th feature, binary)
  Unit test: encode known Pokémon, verify all feature values are correct
  Sanity check: Pokémon with similar roles should have similar embeddings

Step 3 — Showdown client
  Python WebSocket client connecting to Showdown Docker container
  Accepts: your 4 + lead pair, opponent 4 + lead pair
  Returns: winner, turn count, turn 1 HP delta, final Pokémon count each side
  Unit test: play one battle end to end, confirm all return values are populated

Step 4 — Heuristic battle agent
  Implement priority rules within Showdown client
  Unit test: agent makes legal moves every turn, never crashes or times out
  Smoke test: run 100 battles, verify win rates are not 100/0 (agent is functional both sides)

Step 5 — Experiment runner (small scale)
  Implement full loop: pairings → combinations → N battles → store results
  Run on 2 teams × 3 combinations × 5 battles to validate pipeline end to end
  Verify JSON output schema is correct and checkpointing works

Step 6 — Full experiment run
  10 sampled combinations per pairing first (~18,000 battles)
  Verify win rate distribution: combinations should vary meaningfully
  If distribution looks flat (all combinations ~50%) heuristic agent is too random — improve it
  Then run to exhaustion: all 90 combinations per pairing (~162,000 battles total)

Step 7 — Dataset preparation
  Load experiment results, compute bring_score and lead_score labels
  Split: 8 teams train, 2 teams held out
  Build PyTorch Dataset class

Step 8 — MLP baseline
  Train MLP on dataset with dual loss (lead_score primary, bring_score auxiliary)
  Evaluate on held-out teams
  Run sanity checks (team select + lead select scenarios)
  Benchmark vs random combination selection

Step 9 — Transformer upgrade
  Implement Transformer with cross-attention and separate bring/lead pooling heads
  Retrain, verify improvement on held-out performance
  This is the production model

Step 10 — Expand and retrain (Phase 2)
  Add more tournament teams (target 50+)
  Regenerate experiment data
  Retrain model, verify held-out generalisation improves
```

---

## Module Interface

When complete, this module exposes a single clean function:

```python
select_team_and_lead(
  your_roster: List[Pokemon],   # 6 Pokémon with full sets
  opp_roster:  List[Pokemon],   # 6 Pokémon species only
) -> Tuple[List[int], List[int]]
  # (bring_indices, lead_indices)
  # bring_indices: 4 indices into your_roster
  # lead_indices:  2 indices into bring_indices
```

No dependency on the PPO agent, in-battle decisions, or any other module. Takes two rosters of 6, returns which 4 to bring and which 2 to lead. Everything else is internal.

---

## Out of Scope for This Brief

- In-battle decisions (Stage 3) — move/switch/protect per turn
- RL fine-tuning (Phase 3) — requires a general PPO agent first
- Opponent team + lead selection optimisation — opponent uses heuristic during training
- Real-time meta adaptation — Smogon stats pulled once, not live
- Switch decisions mid-battle — handled by Stage 3

---

*Stage 1 & 2: Team Selection + Lead Selection — Implementation Brief | Pokémon VGC AI System*
