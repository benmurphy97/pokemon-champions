import json
import unittest
from pathlib import Path

from champions.stats import compute_stats

FIXTURE = Path(__file__).parent / "fixtures" / "engine_stats.json"


class StatsTest(unittest.TestCase):
    def test_matches_the_engine_for_every_fixture_set(self):
        sets = json.loads(FIXTURE.read_text())
        self.assertEqual(len(sets), 12)
        for s in sets:
            self.assertEqual(compute_stats(s["base"], s["nature"], s["evs"]), s["engine"], s["species"])


if __name__ == "__main__":
    unittest.main()
