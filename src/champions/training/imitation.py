"""Train the policy to imitate the players in the downloaded replays.

    PYTHONPATH=src .venv/bin/python -m champions.training.imitation --epochs 6 --init models/il_policy_v3.pt --preview

Reads cache/raw/*.json.gz (see scrapers/replays.py), turns each game into joint-decision examples
(training/decisions.py), caches them in cache/il_examples.npz, trains PolicyNet and saves the best
model by validation loss. Games are split into train and validation by game, so no game is in both.

Losses (summed):
  * joint action: -log P(a_A, a_B) of what the player did, over the softmax on all legal pairs. A slot
    whose choice the log hides is summed out: -log sum_b P(a_A, b).
  * opponent: cross-entropy of the opponent's actual move or switch for each foe on the field (x0.3).
  * value: squared error between the value head and the game's result for this side, +1 / -1 (x0.25),
    so reinforcement learning starts with a usable baseline.
  * preview (--preview): -log of the total probability of the 90 plans consistent with the replay (x0.3).
"""

import argparse
import gzip
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from champions.model.encoding import ARRAY_KEYS as KEYS
from champions.model.encoding import ENCODING_VERSION, N_ATOMS
from champions.model.model_io import load_policy, save_policy
from champions.model.policy import NEG, PolicyNet
from champions.model.preview import preview_loss
from champions.teams.generate import load_usage
from champions.training.decisions import extract, stack

ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = ROOT / "cache" / "raw"
CACHE = ROOT / "cache" / "il_examples.npz"
MODEL_PATH = ROOT / "models" / "il_policy.pt"


def build_examples(min_rating: int, limit: int | None) -> dict:
    """Extract every recoverable joint decision from the raw logs into stacked arrays."""
    usage = load_usage()
    files = sorted(RAW_DIR.glob("*.json.gz"))[:limit]
    parts, meta = [], {"game": [], "won": [], "rating": []}
    for g, path in enumerate(files):
        replay = json.load(gzip.open(path, "rt"))
        if (replay.get("rating") or 0) < min_rating:
            continue
        lines = replay["log"].split("\n")
        winner = next((l.split("|")[2] for l in lines if l.startswith("|win|")), None)
        names = {l.split("|")[2]: l.split("|")[3] for l in lines if l.startswith("|player|") and l.split("|")[3:4] != [""]}
        examples = extract(lines, usage)
        if not examples:
            continue
        parts.append(stack(examples))
        for e in examples:
            meta["game"].append(g)
            meta["won"].append(1 if names.get(e["side"]) == winner else (-1 if winner else 0))
            meta["rating"].append(replay.get("rating") or 0)
        if (g + 1) % 500 == 0:
            print(f"  extracted {g + 1}/{len(files)} games, {len(meta['game'])} decisions", flush=True)
    data = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    return {**data, **{k: np.array(v) for k, v in meta.items()}}


def load_examples(min_rating: int, limit: int | None, rebuild: bool) -> dict:
    if CACHE.exists() and not rebuild:
        data = dict(np.load(CACHE))
        if int(data.get("encoding_version", 0)) != ENCODING_VERSION:
            raise SystemExit(f"{CACHE.name} was built with encoding version {int(data.get('encoding_version', 0))}, "
                             f"the code is at {ENCODING_VERSION}: rerun with --rebuild")
        return data
    data = build_examples(min_rating, limit)
    CACHE.parent.mkdir(exist_ok=True)
    np.savez_compressed(CACHE, **data, encoding_version=np.array(ENCODING_VERSION))
    return data


def to_batch(data: dict, idx: np.ndarray) -> dict:
    batch = {k: torch.as_tensor(data[k][idx]) for k in KEYS}
    for k in ("label", "foe_label", "won"):
        batch[k] = torch.as_tensor(data[k][idx])
    return batch


def joint_nll(pair_logits: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
    """[B] -log P of the observed atoms; a slot labelled -1 is summed out."""
    logp = torch.log_softmax(pair_logits, dim=-1).reshape(-1, N_ATOMS, N_ATOMS)
    a, b = label[:, 0], label[:, 1]
    rows = torch.arange(len(label))
    by_a = torch.logsumexp(logp, dim=2)[rows, a.clamp(min=0)]          # slot b summed out
    by_b = torch.logsumexp(logp, dim=1)[rows, b.clamp(min=0)]          # slot a summed out
    both = logp[rows, a.clamp(min=0), b.clamp(min=0)]
    return -torch.where((a >= 0) & (b >= 0), both, torch.where(a >= 0, by_a, by_b))


ILLEGAL = 1e6   # a label on a masked option has a loss near 1e9: such labels (about 0.2%, e.g. Struggle or a move Ditto copied) are dropped


def losses(net, batch: dict) -> tuple[torch.Tensor, dict]:
    pair_logits, value, foe_logits = net(batch)
    nll = joint_nll(pair_logits, batch["label"])
    ok = nll < ILLEGAL
    action = nll[ok].mean() if ok.any() else pair_logits.sum() * 0
    foe_label = batch["foe_label"].flatten()
    flat = foe_logits.flatten(0, 1)
    seen = foe_label >= 0
    legal = seen & (flat.gather(1, foe_label.clamp(min=0)[:, None]).squeeze(1) > -ILLEGAL)
    foe = F.cross_entropy(flat[legal], foe_label[legal]) if legal.any() else pair_logits.sum() * 0
    value_loss = ((value - batch["won"].float()) ** 2).mean()
    return action + 0.3 * foe + 0.25 * value_loss, {"action": action.item(), "foe": foe.item(), "value": value_loss.item(),
                                                    "dropped": int((~ok).sum() + (seen & ~legal).sum())}


def evaluate(net, data: dict, idx: np.ndarray, batch_size: int = 512) -> dict:
    """Validation: joint loss, exact-pair accuracy (both slots known), per-slot accuracy, opponent top-1 and top-3."""
    net.eval()
    sums = {"loss": 0.0, "dropped": 0, "pair": 0, "pair_n": 0, "slot": 0, "slot_n": 0, "foe": 0, "foe3": 0, "foe_n": 0, "value": 0.0}
    with torch.no_grad():
        for start in range(0, len(idx), batch_size):
            batch = to_batch(data, idx[start:start + batch_size])
            pair_logits, value, foe_logits = net(batch)
            label = batch["label"]
            nll = joint_nll(pair_logits, label)
            sums["loss"] += nll[nll < ILLEGAL].sum().item(); sums["dropped"] += int((nll >= ILLEGAL).sum())
            best = pair_logits.argmax(-1)
            guess = torch.stack([best // N_ATOMS, best % N_ATOMS], dim=1)
            both = (label >= 0).all(1)
            sums["pair"] += (guess[both] == label[both]).all(1).sum().item(); sums["pair_n"] += both.sum().item()
            known = label >= 0
            sums["slot"] += (guess == label)[known].sum().item(); sums["slot_n"] += known.sum().item()
            fl, fz = batch["foe_label"], foe_logits
            seen = fl >= 0
            top = fz.topk(3, dim=-1).indices
            sums["foe"] += (top[..., 0] == fl)[seen].sum().item()
            sums["foe3"] += (top == fl[..., None]).any(-1)[seen].sum().item(); sums["foe_n"] += seen.sum().item()
            sums["value"] += ((value - batch["won"].float()) ** 2).sum().item()
    n = len(idx)
    return {"loss": sums["loss"] / max(n - sums["dropped"], 1), "dropped": sums["dropped"], "pair": sums["pair"] / max(sums["pair_n"], 1), "slot": sums["slot"] / max(sums["slot_n"], 1),
            "foe": sums["foe"] / max(sums["foe_n"], 1), "foe3": sums["foe3"] / max(sums["foe_n"], 1), "value_mse": sums["value"] / n}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--min-rating", type=int, default=1300)
    parser.add_argument("--limit", type=int, default=None, help="max games to read (with --rebuild)")
    parser.add_argument("--games-fraction", type=float, default=1.0, help="train on this share of the cached games (pilots)")
    parser.add_argument("--rebuild", action="store_true", help="re-extract examples from the raw logs")
    parser.add_argument("--winners-only", action="store_true", help="learn only from the winning player's decisions")
    parser.add_argument("--out", default=str(MODEL_PATH), help="where to save the model")
    parser.add_argument("--preview", action="store_true",
                        help="also learn team preview from cache/preview_examples.npz (built if missing)")
    parser.add_argument("--preview-weight", type=float, default=0.3)
    parser.add_argument("--rating-weight", action="store_true",
                        help="sample stronger players' decisions more often (weight 1 at rating 1350, up to 4 at 1800)")
    parser.add_argument("--init", help="start from this checkpoint (matching weights are carried over, see model_io)")
    parser.add_argument("--dim", type=int, default=192)
    parser.add_argument("--layers", type=int, default=4)
    args = parser.parse_args()

    torch.manual_seed(0); np.random.seed(0)
    torch.set_num_threads(8)
    data = load_examples(args.min_rating, args.limit, args.rebuild)
    games = np.unique(data["game"])
    rng = np.random.default_rng(0)
    if args.games_fraction < 1:
        games = rng.choice(games, size=max(2, int(len(games) * args.games_fraction)), replace=False)
    used = np.isin(data["game"], games)
    held_out = rng.choice(games, size=max(1, len(games) // 10), replace=False)
    is_val = np.isin(data["game"], held_out)
    train_idx = np.flatnonzero(used & ~is_val & ((data["won"] == 1) if args.winners_only else True))
    val_idx = np.flatnonzero(used & is_val)
    print(f"{used.sum()} decisions from {len(games)} games: {len(train_idx)} train, {len(val_idx)} validation", flush=True)

    pv = None
    if args.preview:
        from champions.training.preview_labels import CACHE as PREVIEW_CACHE, build_examples as build_preview
        if not PREVIEW_CACHE.exists() or args.rebuild:
            np.savez_compressed(PREVIEW_CACHE, **build_preview(args.limit, args.min_rating))
        pv = dict(np.load(PREVIEW_CACHE))
        pv_used = np.isin(pv["game"], games)
        pv_val = np.isin(pv["game"], held_out)
        pv_train, pv_val = np.flatnonzero(pv_used & ~pv_val), np.flatnonzero(pv_used & pv_val)
        print(f"{len(pv_train) + len(pv_val)} preview examples: {len(pv_train)} train, {len(pv_val)} validation", flush=True)
    preview_batch = lambda idx: {**{k: torch.as_tensor(pv[k][idx]) for k in KEYS},
                                 "plan_ok": torch.as_tensor(pv["plan_ok"][idx]), "complete": torch.as_tensor(pv["complete"][idx])}

    sample_weight = None
    if args.rating_weight:
        sample_weight = np.clip((data["rating"][train_idx] - 1200) / 150, 0.5, 4.0)
        sample_weight = sample_weight / sample_weight.sum()

    if args.init:
        net, report = load_policy(Path(args.init), report=True)
        print(f"warm start from {args.init}: {report['copied']} tensors copied, {len(report['padded'])} padded, "
              f"{len(report['new'])} new, {len(report['unfit'])} unfit", flush=True)
    else:
        net = PolicyNet(dim=args.dim, layers=args.layers)
    print(f"{sum(p.numel() for p in net.parameters()) / 1e6:.2f}M parameters", flush=True)
    optimiser = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=0.01)
    steps = args.epochs * (len(train_idx) // args.batch_size + 1)
    schedule = torch.optim.lr_scheduler.OneCycleLR(optimiser, max_lr=args.lr, total_steps=steps)

    best = float("inf")
    for epoch in range(1, args.epochs + 1):
        net.train()
        order = rng.choice(train_idx, size=len(train_idx), p=sample_weight) if sample_weight is not None else rng.permutation(train_idx)
        start, total, count, dropped = time.time(), {"action": 0.0, "foe": 0.0, "value": 0.0}, 0, 0
        for i in range(0, len(order), args.batch_size):
            chunk = order[i:i + args.batch_size]
            loss, parts = losses(net, to_batch(data, chunk))
            if pv is not None and len(pv_train):
                loss = loss + args.preview_weight * preview_loss(net, preview_batch(rng.choice(pv_train, 64)))[0]
            optimiser.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimiser.step(); schedule.step()
            for k in total:
                total[k] += parts[k] * len(chunk)
            dropped += parts["dropped"]
            count += len(chunk)
        m = evaluate(net, data, val_idx)
        print(f"epoch {epoch}: train action {total['action'] / count:.3f} foe {total['foe'] / count:.3f} value {total['value'] / count:.3f} | "
              f"val action loss {m['loss']:.3f}  exact pair {m['pair']:.1%}  per slot {m['slot']:.1%}  "
              f"opponent top-1 {m['foe']:.1%} top-3 {m['foe3']:.1%}  value mse {m['value_mse']:.3f}  "
              f"labels dropped: train {dropped}, val {m['dropped']}  ({time.time() - start:.0f}s)", flush=True)
        if pv is not None and len(pv_val):
            net.eval()
            with torch.no_grad():
                pm = preview_loss(net, preview_batch(pv_val))[1]
            print(f"   preview (validation): best plan consistent with the replay {pm['plan'] / pm['n']:.1%}; "
                  f"exact when the four was fully seen {pm['full'] / max(pm['full_n'], 1):.1%} (chance 1.1%)", flush=True)
        if m["loss"] < best:
            best = m["loss"]
            Path(args.out).parent.mkdir(exist_ok=True)
            save_policy(net, Path(args.out), metrics=m, epoch=epoch)
    print(f"best validation action loss {best:.3f} -> {args.out}")


if __name__ == "__main__":
    main()
