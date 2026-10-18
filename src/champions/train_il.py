"""Train the policy to imitate the players in the downloaded replays.

    .venv/bin/python -m champions.train_il --epochs 8

Reads cache/raw/*.json.gz (see import_replays), turns each game into decision examples, caches
them in cache/il_examples.npz, trains PolicyNet, and saves models/il_policy.pt. Games are split
into train and validation by game, so no game appears in both.
"""

import argparse
import gzip
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from champions.decisions import extract
from champions.encoding import ARRAY_KEYS as KEYS
from champions.encoding import ENCODING_VERSION, SWITCH_BASE
from champions.model_io import load_policy, save_policy
from champions.meta import load_usage
from champions.policy import PolicyNet
from champions.vocab import NUM_MOVES

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "cache" / "raw"
CACHE = ROOT / "cache" / "il_examples.npz"
MODEL_PATH = ROOT / "models" / "il_policy.pt"


def build_examples(min_rating: int, limit: int | None) -> dict:
    """Extract every recoverable decision from the raw logs into stacked arrays."""
    usage = load_usage()
    files = sorted(RAW_DIR.glob("*.json.gz"))[:limit]
    arrays: dict[str, list] = {k: [] for k in KEYS}
    meta = {"action": [], "target": [], "game": [], "won": [], "rating": [], "own_tr": []}
    for g, path in enumerate(files):
        replay = json.load(gzip.open(path, "rt"))
        if (replay.get("rating") or 0) < min_rating:
            continue
        lines = replay["log"].split("\n")
        winner = next((l.split("|")[2] for l in lines if l.startswith("|win|")), None)
        names = {l.split("|")[2]: l.split("|")[3] for l in lines if l.startswith("|player|") and l.split("|")[3:4] != [""]}
        for example in extract(lines, usage):
            for k in KEYS:
                arrays[k].append(example["arrays"][k])
            meta["action"].append(example["action"])
            meta["target"].append(example["target"])
            meta["game"].append(g)
            meta["won"].append(int(names.get(example["side"]) == winner))
            meta["rating"].append(replay.get("rating") or 0)
            meta["own_tr"].append(example["own_tr"])
        if (g + 1) % 500 == 0:
            print(f"  extracted {g + 1}/{len(files)} games, {len(meta['action'])} decisions", flush=True)
    return {**{k: np.stack(v) for k, v in arrays.items()}, **{k: np.array(v) for k, v in meta.items()}}


def load_examples(min_rating: int, limit: int | None, rebuild: bool) -> dict:
    if CACHE.exists() and not rebuild:
        data = dict(np.load(CACHE))
        if int(data.get("encoding_version", 0)) != ENCODING_VERSION:
            raise SystemExit(f"{CACHE.name} was built with encoding version {int(data.get('encoding_version', 0))}, "
                             f"the code is at {ENCODING_VERSION}: rerun with --rebuild")
    else:
        data = build_examples(min_rating, limit)
        CACHE.parent.mkdir(exist_ok=True)
        np.savez_compressed(CACHE, **data, encoding_version=np.array(ENCODING_VERSION))
    # The move actually played is always allowed, even if the learnset table did not list it.
    data["mask"][np.arange(len(data["action"])), data["action"]] = True
    return data


def to_batch(data: dict, idx: np.ndarray, device) -> dict:
    return {k: torch.as_tensor(data[k][idx]).to(device) for k in KEYS}


def evaluate(net, data, idx, device, batch_size=1024) -> dict:
    net.eval()
    top1 = top3 = move_ok = switch_ok = n_move = n_switch = 0
    target_ok = n_target = 0
    loss_sum = 0.0
    with torch.no_grad():
        for start in range(0, len(idx), batch_size):
            chunk = idx[start:start + batch_size]
            batch = to_batch(data, chunk, device)
            action = torch.as_tensor(data["action"][chunk]).to(device)
            target = torch.as_tensor(data["target"][chunk]).to(device)
            logits, target_logits = net(batch, target_for_move=action)
            loss_sum += F.cross_entropy(logits, action, reduction="sum").item()
            ranked = logits.topk(3, dim=-1).indices
            top1 += (ranked[:, 0] == action).sum().item()
            top3 += (ranked == action[:, None]).any(-1).sum().item()
            is_switch = action >= SWITCH_BASE
            chose_switch = ranked[:, 0] >= SWITCH_BASE
            switch_ok += (is_switch & chose_switch).sum().item(); n_switch += is_switch.sum().item()
            move_ok += (~is_switch & ~chose_switch).sum().item(); n_move += (~is_switch).sum().item()
            moves = ~is_switch
            target_ok += (target_logits.argmax(-1) == target)[moves].sum().item(); n_target += moves.sum().item()
    n = len(idx)
    return {"loss": loss_sum / n, "top1": top1 / n, "top3": top3 / n,
            "move_vs_switch": (move_ok + switch_ok) / n, "target": target_ok / max(n_target, 1),
            "switch_share": n_switch / n}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--min-rating", type=int, default=1300)
    parser.add_argument("--limit", type=int, default=None, help="max games to read")
    parser.add_argument("--rebuild", action="store_true", help="re-extract examples from the raw logs")
    parser.add_argument("--winners-only", action="store_true", help="learn only from the winning player's decisions")
    parser.add_argument("--tr-weight", type=int, default=1, help="train this many times on decisions by players with a Trick Room team")
    parser.add_argument("--out", default=str(MODEL_PATH), help="where to save the model")
    parser.add_argument("--init", help="start from this checkpoint (weights are adapted if the model has grown)")
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--layers", type=int, default=3)
    args = parser.parse_args()

    torch.manual_seed(0); random.seed(0); np.random.seed(0)
    data = load_examples(args.min_rating, args.limit, args.rebuild)
    n = len(data["action"])
    games = np.unique(data["game"])
    rng = np.random.default_rng(0)
    held_out = set(rng.choice(games, size=max(1, len(games) // 10), replace=False).tolist())
    is_val = np.array([g in held_out for g in data["game"]])
    train_idx = np.flatnonzero(~is_val & ((data["won"] == 1) if args.winners_only else True))
    val_idx = np.flatnonzero(is_val)
    if args.tr_weight > 1:
        extra = train_idx[data["own_tr"][train_idx].astype(bool)]
        train_idx = np.concatenate([train_idx] + [extra] * (args.tr_weight - 1))
        print(f"Trick Room decisions: {len(extra)} ({len(extra) / max(len(train_idx) - (args.tr_weight - 1) * len(extra), 1):.0%} of training), weighted x{args.tr_weight}")
    print(f"{n} decisions from {len(games)} games: {len(train_idx)} train, {len(val_idx)} validation")

    device = torch.device("cpu")
    net = PolicyNet(dim=args.dim, layers=args.layers).to(device)
    if args.init:      # carry an earlier model's weights over (see model_io): new parts start inert
        net, report = load_policy(Path(args.init), report=True)
        print(f"warm start from {args.init}: {report['copied']} tensors copied, {len(report['padded'])} padded, {len(report['new'])} new, {len(report['unfit'])} unfit")
    print(f"{sum(p.numel() for p in net.parameters()) / 1e6:.2f}M parameters")
    optimiser = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=0.01)
    steps = args.epochs * (len(train_idx) // args.batch_size + 1)
    schedule = torch.optim.lr_scheduler.OneCycleLR(optimiser, max_lr=args.lr, total_steps=steps)

    best = 0.0
    for epoch in range(1, args.epochs + 1):
        net.train()
        order = rng.permutation(train_idx)
        start, total, count = time.time(), 0.0, 0
        for i in range(0, len(order), args.batch_size):
            chunk = order[i:i + args.batch_size]
            batch = to_batch(data, chunk, device)
            action = torch.as_tensor(data["action"][chunk]).to(device)
            target = torch.as_tensor(data["target"][chunk]).to(device)
            logits, target_logits = net(batch, target_for_move=action)
            loss = F.cross_entropy(logits, action)
            moves = action < SWITCH_BASE
            if moves.any():
                loss = loss + 0.3 * F.cross_entropy(target_logits[moves], target[moves])
            optimiser.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimiser.step(); schedule.step()
            total += loss.item() * len(chunk); count += len(chunk)
        m = evaluate(net, data, val_idx, device)
        print(f"epoch {epoch}: train loss {total / count:.3f} | val loss {m['loss']:.3f}  top1 {m['top1']:.1%}  "
              f"top3 {m['top3']:.1%}  move/switch {m['move_vs_switch']:.1%}  target {m['target']:.1%}  "
              f"({time.time() - start:.0f}s)", flush=True)
        if m["top1"] > best:
            best = m["top1"]
            Path(args.out).parent.mkdir(exist_ok=True)
            save_policy(net, Path(args.out), metrics=m)

    # Baseline for context: always predict the most common legal action seen in training.
    counts = np.bincount(data["action"][train_idx], minlength=NUM_MOVES + 6)
    legal = data["mask"][val_idx]
    baseline = (np.where(legal, counts[None, :], -1).argmax(1) == data["action"][val_idx]).mean()
    print(f"baseline (most common legal action): top1 {baseline:.1%} | best model top1 {best:.1%}")


if __name__ == "__main__":
    main()
