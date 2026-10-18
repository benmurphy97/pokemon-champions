# Extending the player without starting over

The model is built so most changes keep what it has already learned. What follows is how to make each kind of change, and which ones still need a retrain.

## How old weights survive (`model_io.py`)

A checkpoint stores the network's settings with its weights. `load_policy` rebuilds the network from them and, if the code has grown since, carries the old weights over and starts every new part so it changes nothing:

| Change | What happens to an old checkpoint |
|---|---|
| A new input branch (new `nn.Linear` / `Embedding` added to the query or tokens) | Starts at zero: same play as before until trained. |
| More input columns on an existing layer | Old columns copied, new ones zero. |
| More transformer layers | Extra layers start as the identity. |
| More embedding rows (new kinds of decision, new vocabulary) | Old rows copied, new rows start fresh. |
| A wider network (`dim`) | Not carried over (reported as `unfit`): this one needs a retrain. |

Continue training from an old model with `train_il --init models/old.pt`. `load_policy(path, report=True)` says exactly what was copied, padded or new. `tests/test_model_io.py` checks a deeper model starts out playing identically.

## Recipes

**A derived feature (computed from what the network already sees).** Compute it inside `policy.py` like `damage_features` and `threat_features`. No arrays change, so no re-extraction and no new encoding version. Add its projection as a zero-initialised branch.

**A new input array.** Add it in `encoding.encode` and to `ARRAY_KEYS` (the single list of arrays, used by training, the live agent and the cache), fill it in `decisions.extract` where the label is built, and bump `ENCODING_VERSION`. The training cache refuses to load with the wrong version: rebuild it with `--rebuild` (about 10 minutes for the full 13,900 games), then warm-start from the old checkpoint. Give the new arrays a zero default so existing code paths keep working (see how `partner_action` was added).

**A new head or kind of decision.** Kinds are reserved in `encoding.py`: `PREVIEW_BRING`, `PREVIEW_LEAD` and two spare ones, so adding them does not change the network's shape. A head is a small module that reads the shared trunk output; give it its own loss and data and leave the battle head alone.

**Team preview and lead selection.** The simulator asks Python for the preview choice. `NeuralAgent(net, preview=fn)` takes any function `fn(state, side, request) -> "team 1234"` (the first two numbers lead; the string is the order of the six Pokémon, four of which come). With no function the greedy agent's rule-based preview is used. So you can try a learned preview, a search-based preview, or a hand-written one without touching the battle policy, and compare them with the same arena test. `tests/test_arena.py` shows the hook in use.

**A bigger network.** Depth can be grown in place (extra layers start as identity). Width needs a retrain; keep a run under about 2 hours (see `docs/plan.md`).

**Search at play time.** It does not touch the weights. Wrap `NeuralAgent.decide` (as `match_viewer.Viewer` does) and use the simulator for rollouts.

## Before any full run

Follow `docs/plan.md`: a 5-minute pilot on a tenth of the data, one change at a time, archive the log and model under a run name, and write the result into `docs/learned-agent.md`.
