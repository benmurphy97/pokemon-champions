"""
Stage 5: Genetic Algorithm — DEAP Evolution Engine

Maintains a population of Showdown team chromosomes and evolves them across
generations using DEAP. Each individual is a team string. Fitness is the
meta-weighted win rate from ga/fitness.py.

Architecture decisions (from brief):
  - Island model: separate subpopulations per archetype (Trick Room, Rain, etc.)
  - Fitness sharing / niching: prevents premature convergence within an island
  - Role-aware crossover: swap whole Pokémon rather than individual genes
  - Informed mutation: use ga/chromosome.py mutate() which respects Item Clause
    and Mega Stone constraints

Usage (Phase 3+ — not yet wired into train.py):
    from stage4_team_evolution.evolve import run_ga
    run_ga(
        seed_team=TEAM_TR_ALPHA,
        archetype="trick-room",
        n_generations=20,
        pop_size=10,
        n_battles=5,
        server_cfg=server_cfg,
        model=ppo_model,
        checkpoint_dir="model_outputs/checkpoints/ga",
    )

TODO (Phase 3):
  - Implement run_ga() — wire DEAP toolbox to evaluate_team() via ga/fitness.py
  - Implement island_model() — one DEAP population per archetype, periodic migration
  - Implement crossover() — role-aware swap of whole Pokémon between two teams
  - Parallelise evaluate() calls with Ray (Phase 5)
  - Add Claude API hook for informed mutation suggestions when team has known weaknesses
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# DEAP setup — toolbox is initialised lazily so the module can be imported
# without deap installed during early development phases.
# ---------------------------------------------------------------------------

try:
    from deap import base, creator, tools, algorithms
    _DEAP_AVAILABLE = True
except ImportError:
    _DEAP_AVAILABLE = False
    log.warning(
        "deap not installed — ga/evolve.py is a stub until Phase 3. "
        "Run: pip install deap"
    )


# ---------------------------------------------------------------------------
# Individual and fitness definition
# ---------------------------------------------------------------------------

def _init_deap():
    """Initialise DEAP creator types (idempotent — safe to call multiple times)."""
    if not _DEAP_AVAILABLE:
        raise ImportError("deap is required for Stage 5. Run: pip install deap")

    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "Individual"):
        # An Individual is a Python string (Showdown team format) wrapped in a list
        # so DEAP's list-based operators work on the population container.
        creator.create("Individual", list, fitness=creator.FitnessMax)


# ---------------------------------------------------------------------------
# Crossover — role-aware Pokémon swap
# ---------------------------------------------------------------------------

def crossover(ind1: list, ind2: list) -> tuple[list, list]:
    """
    Role-aware crossover: swap one non-Mega Pokémon between two team strings.

    TODO: Implement role-aware selection — prefer swapping Pokémon with the same
    role tag (e.g. "rain setter" for "rain setter") to preserve archetype coherence.
    TODO: Validate item clause and mega constraint after swap; reject and return
    originals unchanged if constraints are violated.

    Args:
        ind1, ind2: DEAP Individual objects (single-element lists containing a
                    Showdown team string each).

    Returns:
        Modified (ind1, ind2) pair.
    """
    from stage4_team_evolution.chromosome import parse_team, serialise_team, EV_STATS
    from stage4_team_evolution.pool import MEGA_STONES
    import random

    team1 = parse_team(ind1[0])
    team2 = parse_team(ind2[0])

    # Only swap non-Mega holders to preserve archetype cores
    swappable1 = [i for i, m in enumerate(team1) if not (m.get("item", "") or "").endswith("ite")]
    swappable2 = [i for i, m in enumerate(team2) if not (m.get("item", "") or "").endswith("ite")]

    if not swappable1 or not swappable2:
        log.debug("crossover: no swappable slots — returning unchanged")
        return ind1, ind2

    # TODO: role-aware selection instead of random
    i1 = random.choice(swappable1)
    i2 = random.choice(swappable2)

    # Tentative swap
    team1[i1], team2[i2] = team2[i2], team1[i1]

    # TODO: validate item clause and mega constraint; revert on violation
    ind1[0] = serialise_team(team1)
    ind2[0] = serialise_team(team2)

    del ind1.fitness.values
    del ind2.fitness.values
    return ind1, ind2


# ---------------------------------------------------------------------------
# Mutation wrapper
# ---------------------------------------------------------------------------

def mutate_individual(ind: list, tenure: dict | None = None) -> tuple[list]:
    """DEAP-compatible mutation operator wrapping ga/chromosome.mutate()."""
    from stage4_team_evolution.chromosome import mutate

    ind[0] = mutate(ind[0], tenure=tenure)
    del ind.fitness.values
    return (ind,)


# ---------------------------------------------------------------------------
# Fitness evaluation
# ---------------------------------------------------------------------------

def evaluate_individual(ind: list, model, server_cfg, n_battles: int = 5) -> tuple[float]:
    """
    DEAP evaluate() callback. Returns a 1-tuple (fitness,) as required by DEAP.

    TODO: Cache fitness values between generations to avoid re-evaluating
    identical team strings (hash-based deduplication).
    TODO: Parallelise with Ray.remote() for Phase 5.
    """
    from stage4_team_evolution.fitness import evaluate_team

    team_str = ind[0]
    try:
        fitness, _ = evaluate_team(model, server_cfg, team_str, n_battles=n_battles)
    except Exception as exc:
        log.warning("evaluate_individual failed: %s — assigning fitness=0", exc)
        fitness = 0.0
    return (fitness,)


# ---------------------------------------------------------------------------
# Single-island GA run
# ---------------------------------------------------------------------------

def run_ga(
    seed_team: str,
    archetype: str,
    n_generations: int,
    pop_size: int,
    n_battles: int,
    server_cfg,
    model,
    checkpoint_dir: str = "model_outputs/checkpoints/ga",
    crossover_prob: float = 0.5,
    mutation_prob: float = 0.3,
    snapshot_every: int = 5,
) -> str:
    """
    Run one island of the GA for a given archetype. Returns the best team string.

    TODO: Implement fully — currently a skeleton.
    TODO: Add niching / fitness sharing to maintain diversity within the island.
    TODO: Save generation-over-generation deltas to MLflow / W&B (Phase 5).
    TODO: Add meta-game scout hook — pull current Showdown ladder usage and
          adjust archetype weights between generations.

    Args:
        seed_team:       Showdown team string used to seed the initial population.
        archetype:       Human-readable archetype tag (e.g. "trick-room", "rain").
        n_generations:   Number of GA generations to run.
        pop_size:        Population size per island.
        n_battles:       Battles per team per fitness evaluation.
        server_cfg:      Showdown server configuration.
        model:           Loaded MaskablePPO model for battle evaluation.
        checkpoint_dir:  Directory to save best team per generation.
        crossover_prob:  Probability of crossover between two individuals.
        mutation_prob:   Probability of mutation per individual.
        snapshot_every:  Save a snapshot every N generations.

    Returns:
        Best team string found across all generations.
    """
    if not _DEAP_AVAILABLE:
        raise ImportError("deap is required for Stage 5. Run: pip install deap")

    _init_deap()

    from deap import base, tools, algorithms
    import random
    from stage4_team_evolution.chromosome import mutate

    toolbox = base.Toolbox()

    # Individual factory — start from seed_team and apply one mutation
    def make_individual():
        team = mutate(seed_team) if pop_size > 1 else seed_team
        return creator.Individual([team])

    toolbox.register("individual", make_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register(
        "evaluate",
        evaluate_individual,
        model=model,
        server_cfg=server_cfg,
        n_battles=n_battles,
    )
    toolbox.register("mate", crossover)
    toolbox.register("mutate", mutate_individual)
    toolbox.register("select", tools.selTournament, tournsize=3)

    pop = toolbox.population(n=pop_size)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", lambda x: sum(v[0] for v in x) / max(1, len(x)))
    stats.register("max", lambda x: max(v[0] for v in x))

    hall_of_fame = tools.HallOfFame(1)

    log.info("GA starting: archetype=%s  pop=%d  generations=%d", archetype, pop_size, n_generations)

    # TODO: replace with eaSimple once crossover / mutation are fully implemented.
    # For now, evaluate the initial population only (stub behaviour).
    for ind in pop:
        if not ind.fitness.valid:
            ind.fitness.values = toolbox.evaluate(ind)

    hall_of_fame.update(pop)
    record = stats.compile(pop)
    log.info("Gen 0 — avg=%.1f%%  max=%.1f%%", record["avg"] * 100, record["max"] * 100)

    # TODO: implement full generational loop with crossover, mutation, selection.
    # Pseudocode for reference:
    #
    # for gen in range(1, n_generations + 1):
    #     offspring = toolbox.select(pop, len(pop))
    #     offspring = algorithms.varAnd(offspring, toolbox, crossover_prob, mutation_prob)
    #     invalid = [ind for ind in offspring if not ind.fitness.valid]
    #     for ind in invalid:
    #         ind.fitness.values = toolbox.evaluate(ind)
    #     pop[:] = offspring
    #     hall_of_fame.update(pop)
    #     if gen % snapshot_every == 0:
    #         _save_snapshot(hall_of_fame[0][0], archetype, gen, checkpoint_dir)

    best_team = hall_of_fame[0][0]
    log.info("GA complete. Best fitness: %.1f%%", hall_of_fame[0].fitness.values[0] * 100)
    return best_team


# ---------------------------------------------------------------------------
# Island model (multi-archetype)
# ---------------------------------------------------------------------------

def island_model(
    seed_teams: dict[str, str],
    n_generations: int,
    pop_size: int,
    n_battles: int,
    server_cfg,
    model,
    checkpoint_dir: str = "model_outputs/checkpoints/ga",
    migration_every: int = 5,
    migration_size: int = 1,
) -> dict[str, str]:
    """
    Run one GA island per archetype and return the best team per archetype.

    TODO: Implement periodic migration — every migration_every generations,
    copy migration_size best individuals from each island to a random neighbour.
    This introduces diversity without collapsing all islands to one solution.

    TODO: Allocate more population budget to archetypes with higher meta usage
    (from data/scraper Pikalytics data).

    Args:
        seed_teams:      {archetype: seed_team_string}
        migration_every: How often to migrate individuals between islands.
        migration_size:  How many individuals to migrate per island per migration.

    Returns:
        {archetype: best_team_string}
    """
    # TODO: implement; currently returns seed teams unchanged as a stub.
    log.warning(
        "island_model: multi-archetype GA not yet implemented — returning seed teams unchanged. "
        "This is a Phase 4 task."
    )
    return dict(seed_teams)


# ---------------------------------------------------------------------------
# Snapshot helpers
# ---------------------------------------------------------------------------

def _save_snapshot(team_str: str, archetype: str, generation: int, checkpoint_dir: str):
    """Save the best team at this generation to disk."""
    import os
    path = Path(checkpoint_dir) / archetype
    path.mkdir(parents=True, exist_ok=True)
    out = path / f"gen{generation:04d}_best.txt"
    out.write_text(team_str)
    log.info("GA snapshot → %s", out)
