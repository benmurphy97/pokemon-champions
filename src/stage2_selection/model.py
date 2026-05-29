"""
SelectionMLP — scores (bring, lead) combinations for team selection.

Architecture:
  Per-pokemon encoder (shared weights within group) → mean pool → MLP head

  Our mon:  embed 7 IDs (embed_dim each) → cat floats (90)  → Linear → ReLU → (hidden,)
  Opp mon:  embed 1 ID                   → cat floats (148) → Linear → ReLU → (hidden,)
  our_team  = mean over 4 our  mon vecs
  opp_team  = mean over 6 opp  mon vecs
  head:     Linear(hidden*2 → hidden) → ReLU → Linear(hidden → 1)

Inference:
  predict_best() scores all 90 (bring, lead) combinations and returns the top one.

Usage:
    docker-compose run --rm trainer python src/selection/model.py --epochs 20
"""
from __future__ import annotations

import argparse
import random
from itertools import combinations
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

_PROJECT_ROOT = Path(__file__).parent.parent.parent
_DATA_DIR   = _PROJECT_ROOT / "data"
_TEAMS_DIR  = _DATA_DIR / "teams"
_CKPT_DIR   = _PROJECT_ROOT / "model_outputs" / "cache"
_CKPT_PATH  = _CKPT_DIR / "selection_model.pt"
_VOCAB_PATH = _CKPT_DIR / "vocab.json"
_JSONL_PATH = _DATA_DIR / "experiment_results.jsonl"


# ── Model ────────────────────────────────────────────────────────────────────

class SelectionMLP(nn.Module):
    """
    Args:
        vocab_sizes: dict with keys species, abilities, items, moves — vocab lengths
        embed_dim:   embedding dimension per ID
        hidden_dim:  hidden layer width
    """

    YOUR_FLOAT = 90
    OPP_FLOAT  = 148
    YOUR_IDS   = 7   # species, ability, item, move×4
    OPP_IDS    = 1   # species only

    def __init__(
        self,
        vocab_sizes: dict[str, int],
        embed_dim:  int = 32,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()

        self.species_embed  = nn.Embedding(vocab_sizes["species"],   embed_dim, padding_idx=0)
        self.ability_embed  = nn.Embedding(vocab_sizes["abilities"], embed_dim, padding_idx=0)
        self.item_embed     = nn.Embedding(vocab_sizes["items"],     embed_dim, padding_idx=0)
        self.move_embed     = nn.Embedding(vocab_sizes["moves"],     embed_dim, padding_idx=0)

        # Our mon: 7 IDs × embed_dim + 90 floats → hidden
        our_in = self.YOUR_IDS * embed_dim + self.YOUR_FLOAT
        self.our_enc = nn.Sequential(
            nn.Linear(our_in, hidden_dim),
            nn.ReLU(),
        )

        # Opp mon: 1 ID × embed_dim + 148 floats → hidden
        opp_in = self.OPP_IDS * embed_dim + self.OPP_FLOAT
        self.opp_enc = nn.Sequential(
            nn.Linear(opp_in, hidden_dim),
            nn.ReLU(),
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def _embed_our(self, ids: torch.Tensor) -> torch.Tensor:
        """Embed all 7 our-mon ID slots (species, ability, item, move×4) and flatten to (..., 7E)."""
        sp  = self.species_embed(ids[..., 0])    # (..., E)
        ab  = self.ability_embed(ids[..., 1])
        it  = self.item_embed(ids[..., 2])
        mv  = self.move_embed(ids[..., 3:])      # (..., 4, E)
        mv  = mv.flatten(-2, -1)                 # (..., 4E)
        return torch.cat([sp, ab, it, mv], dim=-1)  # (..., 7E)

    def forward(
        self,
        our_ids:    torch.Tensor,  # (B, 4, 7)
        our_floats: torch.Tensor,  # (B, 4, 90)
        opp_ids:    torch.Tensor,  # (B, 6, 1)
        opp_floats: torch.Tensor,  # (B, 6, 148)
    ) -> torch.Tensor:             # (B,)
        # Our team
        our_emb   = self._embed_our(our_ids)                         # (B, 4, 7E)
        our_input = torch.cat([our_emb, our_floats], dim=-1)         # (B, 4, 7E+90)
        our_vecs  = self.our_enc(our_input)                          # (B, 4, H)
        our_team  = our_vecs.mean(dim=1)                             # (B, H)

        # Opp team
        opp_emb   = self.species_embed(opp_ids.squeeze(-1))          # (B, 6, E)
        opp_input = torch.cat([opp_emb, opp_floats], dim=-1)         # (B, 6, E+148)
        opp_vecs  = self.opp_enc(opp_input)                          # (B, 6, H)
        opp_team  = opp_vecs.mean(dim=1)                             # (B, H)

        combined = torch.cat([our_team, opp_team], dim=-1)           # (B, 2H)
        return self.head(combined).squeeze(-1)                        # (B,)


# ── Inference helper ─────────────────────────────────────────────────────────

def predict_best(
    model:      SelectionMLP,
    our_team,                  # selection.pokemon.Team
    opp_team,                  # selection.pokemon.Team
    encoder,                   # selection.encoder.PokemonEncoder
    device:     str = "cpu",
) -> tuple[list[int], list[int]]:
    """Score all 90 (bring, lead) combos; return (bring_indices, lead_indices) with highest score."""
    model.eval()

    opp_enc = [encoder.encode_opp_pokemon(m.species) for m in opp_team.pokemon]
    opp_ids_t    = torch.tensor([[e["ids"]    for e in opp_enc]], dtype=torch.long,    device=device)
    opp_floats_t = torch.tensor([[e["floats"] for e in opp_enc]], dtype=torch.float32, device=device)

    our_base = [encoder.encode_your_pokemon(m, False) for m in our_team.pokemon]

    best_score  = float("-inf")
    best_bring  = [0, 1, 2, 3]
    best_lead   = [0, 1]

    with torch.no_grad():
        for bring in combinations(range(6), 4):
            for lead in combinations(range(4), 2):
                lead_set   = set(lead)
                our_ids_l, our_floats_l = [], []
                for slot, bi in enumerate(bring):
                    e      = our_base[bi]
                    floats = list(e["floats"])
                    floats[-1] = 1.0 if slot in lead_set else 0.0
                    our_ids_l.append(e["ids"])
                    our_floats_l.append(floats)

                our_ids_t    = torch.tensor([our_ids_l],    dtype=torch.long,    device=device)
                our_floats_t = torch.tensor([our_floats_l], dtype=torch.float32, device=device)

                score = model(our_ids_t, our_floats_t, opp_ids_t, opp_floats_t).item()
                if score > best_score:
                    best_score = score
                    best_bring = list(bring)
                    best_lead  = list(lead)

    return best_bring, best_lead


# ── Training ─────────────────────────────────────────────────────────────────

def _train(args) -> None:
    """Train SelectionMLP on experiment JSONL data; saves best val-loss checkpoint."""
    from stage2_selection.dataset import SelectionDataset

    _CKPT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading dataset …")
    ds = SelectionDataset(
        jsonl_path=args.jsonl,
        teams_dir=args.teams_dir,
        data_dir=args.data_dir,
        vocab_path=str(_VOCAB_PATH),
    )
    print(f"  {len(ds)} samples")

    # Split by pairing to keep val pairings unseen
    pairings = list({(r["our_roster"], r["opp_roster"]) for r in ds.records})
    random.seed(42)
    random.shuffle(pairings)
    n_val    = max(1, len(pairings) // 5)
    val_set  = set(pairings[:n_val])

    train_idx = [i for i, r in enumerate(ds.records)
                 if (r["our_roster"], r["opp_roster"]) not in val_set]
    val_idx   = [i for i, r in enumerate(ds.records)
                 if (r["our_roster"], r["opp_roster"]) in val_set]

    train_loader = DataLoader(Subset(ds, train_idx), batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(Subset(ds, val_idx),   batch_size=args.batch_size)

    vocab_sizes = {
        "species":   len(ds.vocab.species),
        "abilities": len(ds.vocab.abilities),
        "items":     len(ds.vocab.items),
        "moves":     len(ds.vocab.moves),
    }
    print(f"  vocab: {vocab_sizes}")
    print(f"  train={len(train_idx)}  val={len(val_idx)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = SelectionMLP(vocab_sizes, embed_dim=args.embed_dim, hidden_dim=args.hidden_dim).to(device)
    opt    = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            our_ids    = batch["our_ids"].to(device)
            our_floats = batch["our_floats"].to(device)
            opp_ids    = batch["opp_ids"].to(device)
            opp_floats = batch["opp_floats"].to(device)
            scores     = batch["score"].to(device)

            pred = model(our_ids, our_floats, opp_ids, opp_floats)
            loss = loss_fn(pred, scores)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss += loss.item() * len(scores)

        train_loss /= len(train_idx)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                pred = model(
                    batch["our_ids"].to(device),
                    batch["our_floats"].to(device),
                    batch["opp_ids"].to(device),
                    batch["opp_floats"].to(device),
                )
                val_loss += loss_fn(pred, batch["score"].to(device)).item() * len(batch["score"])
        val_loss /= max(len(val_idx), 1)

        flag = ""
        if val_loss < best_val:
            best_val = val_loss
            torch.save({"model_state": model.state_dict(), "vocab_sizes": vocab_sizes,
                        "embed_dim": args.embed_dim, "hidden_dim": args.hidden_dim}, _CKPT_PATH)
            flag = "  ✓ saved"

        print(f"epoch {epoch:3d}  train={train_loss:.4f}  val={val_loss:.4f}{flag}")

    print(f"\nBest val loss: {best_val:.4f}  →  {_CKPT_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl",      default=str(_JSONL_PATH))
    parser.add_argument("--teams-dir",  default=str(_TEAMS_DIR))
    parser.add_argument("--data-dir",   default=str(_DATA_DIR))
    parser.add_argument("--epochs",     type=int,   default=20)
    parser.add_argument("--batch-size", type=int,   default=64)
    parser.add_argument("--embed-dim",  type=int,   default=32)
    parser.add_argument("--hidden-dim", type=int,   default=256)
    parser.add_argument("--lr",         type=float, default=1e-3)
    args = parser.parse_args()
    _train(args)
