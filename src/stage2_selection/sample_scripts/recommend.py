"""
Team selection recommendation: Wolfe vs. TR Alpha.

Loads the trained SelectionMLP and scores all 90 (bring, lead) combinations
for Wolfe's team against the TR Alpha opponent. Prints the top picks with a
plain-English breakdown of each recommendation.

Usage (from project root):
    docker-compose run --rm trainer python src/selection/sample_scripts/recommend.py

    # Optional flags:
    #   --top N          show top-N combinations   (default: 5)
    #   --our-team PATH  override your team file
    #   --opp-team PATH  override opponent team file
"""
from __future__ import annotations

import argparse
import sys
from itertools import combinations
from pathlib import Path

import torch

# ── Path setup ────────────────────────────────────────────────────────────────
_SRC = Path(__file__).parent.parent.parent          # .../src/
_ROOT = _SRC.parent                                  # project root
sys.path.insert(0, str(_SRC))

from stage2_selection.encoder import PokemonEncoder
from stage2_selection.model import SelectionMLP
from stage2_selection.pokemon import load_team_from_file

_CKPT_PATH  = _ROOT / "model_outputs" / "cache" / "selection_model.pt"
_VOCAB_PATH = _ROOT / "model_outputs" / "cache" / "vocab.json"
_DATA_DIR   = _ROOT / "data"
_TEAMS_DIR  = _DATA_DIR / "teams"

# Role tag labels — must match encoder._role_tags() order
_ROLE_LABELS = [
    "TR setter",
    "TR abuser",
    "weather",
    "terrain",
    "redirector",
    "spread attacker",
    "protect",
    "bulky",
    "fast",
    "speed control",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_model() -> tuple[SelectionMLP, dict]:
    ckpt = torch.load(_CKPT_PATH, map_location="cpu", weights_only=False)
    model = SelectionMLP(
        vocab_sizes=ckpt["vocab_sizes"],
        embed_dim=ckpt.get("embed_dim", 32),
        hidden_dim=ckpt.get("hidden_dim", 256),
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt["vocab_sizes"]


def _score_all(model, our_team, opp_team, encoder, top_n: int):
    """Score every (bring, lead) combination and return the top_n, best first."""
    opp_enc = [encoder.encode_opp_pokemon(m.species) for m in opp_team.pokemon]
    opp_ids_t    = torch.tensor([[e["ids"]    for e in opp_enc]], dtype=torch.long)
    opp_floats_t = torch.tensor([[e["floats"] for e in opp_enc]], dtype=torch.float32)

    our_base = [encoder.encode_your_pokemon(m, False) for m in our_team.pokemon]

    results = []
    with torch.no_grad():
        for bring in combinations(range(len(our_team.pokemon)), 4):
            for lead in combinations(range(4), 2):
                lead_set = set(lead)
                our_ids_l, our_floats_l = [], []
                for slot, bi in enumerate(bring):
                    e = our_base[bi]
                    floats = list(e["floats"])
                    floats[-1] = 1.0 if slot in lead_set else 0.0
                    our_ids_l.append(e["ids"])
                    our_floats_l.append(floats)

                our_ids_t    = torch.tensor([our_ids_l],    dtype=torch.long)
                our_floats_t = torch.tensor([our_floats_l], dtype=torch.float32)
                score = model(our_ids_t, our_floats_t, opp_ids_t, opp_floats_t).item()
                results.append((score, list(bring), list(lead)))

    results.sort(key=lambda x: x[0], reverse=True)
    return results[:top_n]


def _role_tags_for(mon, encoder) -> list[str]:
    """Return list of active role-tag labels for a Pokemon."""
    enc = encoder.encode_your_pokemon(mon, False)
    # role tags occupy floats[-11:-1] (indices 79-88 of 90)
    tag_floats = enc["floats"][79:89]
    return [label for label, v in zip(_ROLE_LABELS, tag_floats) if v > 0.0]


def _print_banner(title: str) -> None:
    width = 62
    print()
    print("=" * width)
    print(f"  {title}")
    print("=" * width)


def _print_team_summary(team, label: str, encoder) -> None:
    print(f"\n{label} ({team.name})")
    print("-" * 50)
    for i, mon in enumerate(team.pokemon):
        tags = _role_tags_for(mon, encoder)
        tag_str = ", ".join(tags) if tags else "—"
        print(f"  [{i}] {mon.species:<20} {mon.item:<20}  [{tag_str}]")


def _describe_bring(our_team, bring, lead, encoder, score: float, rank: int) -> None:
    lead_slots = set(lead)
    bring_mons = [our_team.pokemon[i] for i in bring]

    if rank == 1:
        print(f"\n★  RECOMMENDED SELECTION  (model score: {score:+.4f})")
    else:
        print(f"\n#{rank}  Alternative  (model score: {score:+.4f})")

    print()
    print("  BRING (4 Pokemon):")
    for slot, (bi, mon) in enumerate(zip(bring, bring_mons)):
        lead_marker = " ← LEAD" if slot in lead_slots else ""
        tags = _role_tags_for(mon, encoder)
        tag_str = ", ".join(tags) if tags else "none"
        print(f"    Slot {slot}  [{bi}] {mon.species:<20}  roles: {tag_str}{lead_marker}")

    lead_mons = [bring_mons[s] for s in lead]
    print()
    print(f"  LEAD (2 Pokemon in front):")
    for mon in lead_mons:
        print(f"    • {mon.species}")

    # Narrative
    print()
    print("  Why the model favors this:")
    _print_narrative(bring_mons, lead_mons, [our_team.pokemon[i] for i in bring], lead_slots)


def _print_narrative(bring_mons, lead_mons, _, lead_slots) -> None:
    all_roles: dict[str, list[str]] = {}
    for mon in bring_mons:
        # recompute tags directly from moves/ability using the role-tag names
        pass

    # Gather role flags from bring
    tr_setters   = [m.species for m in bring_mons if "Trick Room" in m.moves]
    tr_abusers   = [m.species for m in bring_mons
                    if m.base_stats.get("spe", 999) <= 60 and m.species not in tr_setters]
    redirectors  = [m.species for m in bring_mons
                    if any(mv in {"Follow Me", "Rage Powder"} for mv in m.moves)]
    fake_outers  = [m.species for m in bring_mons if "Fake Out" in m.moves]
    speed_ctrl   = [m.species for m in bring_mons
                    if any(mv in {"Tailwind", "Icy Wind", "Electroweb", "Thunder Wave"} for mv in m.moves)]
    wide_guard   = [m.species for m in bring_mons if "Wide Guard" in m.moves]

    lead_names = [m.species for m in lead_mons]

    lines = []

    if tr_setters:
        lines.append(
            f"Trick Room is available via {', '.join(tr_setters)}, "
            f"shaping the team around reversed speed priority."
        )
    if tr_abusers:
        lines.append(
            f"{', '.join(tr_abusers)} can exploit TR with low Speed stats."
        )
    if fake_outers:
        lines.append(
            f"{', '.join(fake_outers)} provide Fake Out pressure to buy a free turn."
        )
    if redirectors:
        lines.append(
            f"{', '.join(redirectors)} can redirect single-target attacks to protect key threats."
        )
    if speed_ctrl:
        lines.append(
            f"{', '.join(speed_ctrl)} offer speed control (Tailwind / Icy Wind) to offset the opponent's pace."
        )
    if wide_guard:
        lines.append(
            f"{', '.join(wide_guard)} has Wide Guard to wall the opponent's spread moves."
        )

    lines.append(
        f"The lead of {lead_names[0]} + {lead_names[1]} was rated highest because "
        f"the model expects them to apply the most immediate pressure against this opponent."
    )

    for line in lines:
        # Word-wrap at ~56 chars
        words = line.split()
        current, out = [], []
        for word in words:
            if sum(len(w) + 1 for w in current) + len(word) > 56:
                out.append("    " + " ".join(current))
                current = [word]
            else:
                current.append(word)
        if current:
            out.append("    " + " ".join(current))
        print("\n".join(out))


def _print_opp_threats(opp_team, encoder) -> None:
    print("\n  Opponent threats to watch:")
    for mon in opp_team.pokemon:
        enc = encoder.encode_opp_pokemon(mon.species)
        tag_floats = enc["floats"][138:148]  # last 10 of 148 OPP floats = role tags
        tags = [label for label, v in zip(_ROLE_LABELS, tag_floats) if v > 0.0]
        tag_str = ", ".join(tags) if tags else "none flagged"
        print(f"    • {mon.species:<20}  [{tag_str}]")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--top",      type=int, default=5, help="show top-N combos")
    parser.add_argument("--our-team", default=str(_TEAMS_DIR / "wolfe.txt"),
                        help="your team file (Showdown paste .txt)")
    parser.add_argument("--opp-team", default=str(_TEAMS_DIR / "tr_alpha.txt"),
                        help="opponent team file (Showdown paste .txt)")
    args = parser.parse_args()

    print("\nLoading model checkpoint …")
    model, vocab_sizes = _load_model()
    print(f"  vocab sizes: {vocab_sizes}")

    print("Loading encoder …")
    encoder = PokemonEncoder.from_data_dir(
        data_dir=_DATA_DIR,
        teams_dir=_TEAMS_DIR,
        vocab_path=str(_VOCAB_PATH),
    )

    print("Loading teams …")
    our_team = load_team_from_file(args.our_team)
    opp_team = load_team_from_file(args.opp_team)
    print(f"  Our team  : {our_team.name}  ({len(our_team.pokemon)} Pokemon)")
    print(f"  Opp team  : {opp_team.name}  ({len(opp_team.pokemon)} Pokemon)")

    # Attach base_stats from encoder data so narrative helpers can read them
    for mon in our_team.pokemon:
        d = encoder._details.get(mon.species) or {}
        mon.base_stats = d.get("base_stats", {stat: 0 for stat in ["hp","atk","def","spa","spd","spe"]})
    for mon in opp_team.pokemon:
        d = encoder._details.get(mon.species) or {}
        mon.base_stats = d.get("base_stats", {stat: 0 for stat in ["hp","atk","def","spa","spd","spe"]})

    _print_banner(f"MATCHUP: {our_team.name.upper()} vs. {opp_team.name.upper()}")

    _print_team_summary(our_team, "Your team", encoder)
    _print_team_summary(opp_team, "Opponent team", encoder)

    print("\nScoring all 90 (bring × lead) combinations …")
    top = _score_all(model, our_team, opp_team, encoder, top_n=args.top)

    _print_banner("RECOMMENDATIONS")

    for rank, (score, bring, lead) in enumerate(top, start=1):
        _describe_bring(our_team, bring, lead, encoder, score, rank)
        if rank == 1:
            _print_opp_threats(opp_team, encoder)

    print()
    print("=" * 62)
    print("  Run with --top N to see more or fewer alternatives.")
    print("=" * 62)
    print()


if __name__ == "__main__":
    main()
