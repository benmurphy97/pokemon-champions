"""
SelectionDataset — PyTorch Dataset over experiment_results.jsonl.

Each sample represents one (our_team × bring_combo × lead_combo × opp_team)
experiment outcome. Returns encoded tensors + a scalar score label.

Label: win_rate + 0.3 × avg_turn1_hp_delta
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

from selection.encoder import PokemonEncoder
from selection.pokemon import load_all_teams


def _load_jsonl(path: Path) -> list[dict]:
    """Read all non-empty lines from a JSONL file into a list of dicts."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


class SelectionDataset(Dataset):
    """
    Args:
        jsonl_path:  Path to experiment_results.jsonl
        teams_dir:   Directory containing *.txt Showdown team files
        data_dir:    Data directory (champions_pokemon_details.json etc.)
        vocab_path:  Optional path to cached vocab.json (built on first run)
    """

    def __init__(
        self,
        jsonl_path: str | Path,
        teams_dir:  str | Path,
        data_dir:   str | Path,
        vocab_path: str | Path | None = None,
    ) -> None:
        self.records = _load_jsonl(Path(jsonl_path))

        teams      = load_all_teams(Path(teams_dir))
        encoder    = PokemonEncoder.from_data_dir(data_dir, vocab_path=vocab_path)
        self.vocab = encoder.vocab

        # Pre-encode all 6 mons per team (is_lead=False; lead flag set per-sample)
        self._our: dict[str, list[dict]] = {
            t.name: [encoder.encode_your_pokemon(m, False) for m in t.pokemon]
            for t in teams
        }
        self._opp: dict[str, list[dict]] = {
            t.name: [encoder.encode_opp_pokemon(m.species) for m in t.pokemon]
            for t in teams
        }

    def __len__(self) -> int:
        """Number of (bring, lead, matchup) samples."""
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Return encoded tensors and score label for one experiment record."""
        rec   = self.records[idx]
        bring = rec["bring_indices"]       # list[int], len 4
        leads = set(rec["lead_indices"])   # set of 2 indices into bring

        # Our team: 4 brought mons, lead flag flipped for the 2 leads
        our_enc = self._our[rec["our_roster"]]
        our_ids, our_floats = [], []
        for slot, bi in enumerate(bring):
            e      = our_enc[bi]
            floats = list(e["floats"])          # copy — don't mutate pre-computed
            floats[-1] = 1.0 if slot in leads else 0.0
            our_ids.append(e["ids"])
            our_floats.append(floats)

        # Opp team: all 6 mons
        opp_enc    = self._opp[rec["opp_roster"]]
        opp_ids    = [e["ids"]    for e in opp_enc]
        opp_floats = [e["floats"] for e in opp_enc]

        score = rec["win_rate"] + 0.3 * rec["avg_turn1_hp_delta"]

        return {
            "our_ids":    torch.tensor(our_ids,    dtype=torch.long),     # (4, 7)
            "our_floats": torch.tensor(our_floats, dtype=torch.float32),  # (4, 90)
            "opp_ids":    torch.tensor(opp_ids,    dtype=torch.long),     # (6, 1)
            "opp_floats": torch.tensor(opp_floats, dtype=torch.float32),  # (6, 148)
            "score":      torch.tensor(score,      dtype=torch.float32),  # ()
        }
