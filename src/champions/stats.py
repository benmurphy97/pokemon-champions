"""Battle stats under the Champions rules, from base stats, a nature and stat points.

HP = base + points + 75; every other stat = base + points + 20, then the nature adds 10% or takes 10%
(truncated). Checked against the engine's own numbers in tests/test_stats.py.
"""

import json

from champions.vocab import REFERENCE

STAT_KEYS = ("hp", "atk", "def", "spa", "spd", "spe")
_NATURES = json.loads((REFERENCE / "natures.json").read_text())


def compute_stats(base: dict, nature: str, points: dict) -> dict:
    """`base` and `points` are keyed hp/atk/def/spa/spd/spe; `points` may omit stats (treated as 0)."""
    plus, minus = _NATURES.get(nature, {}).get("plus"), _NATURES.get(nature, {}).get("minus")
    stats = {"hp": base["hp"] + points.get("hp", 0) + 75}
    for key in STAT_KEYS[1:]:
        value = base[key] + points.get(key, 0) + 20
        if key == plus:
            value = value * 110 // 100
        elif key == minus:
            value = value * 90 // 100
        stats[key] = value
    return stats
