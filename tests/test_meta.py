import unittest

from champions.calibrate import ranks, spearman
from champions.meta import generate_pool, load_usage
from champions.simulate import RUNNER

SHOWDOWN_BUILT = (RUNNER.parent.parent / "vendor" / "pokemon-showdown" / "dist" / "sim").exists()


class CalibrateTest(unittest.TestCase):
    def test_ranks_share_ties(self):
        self.assertEqual(ranks([10, 20, 20, 30]), [1.0, 2.5, 2.5, 4.0])

    def test_spearman_extremes(self):
        self.assertAlmostEqual(spearman([1, 2, 3, 4], [10, 20, 30, 40]), 1.0)
        self.assertAlmostEqual(spearman([1, 2, 3, 4], [40, 30, 20, 10]), -1.0)


class UsageDataTest(unittest.TestCase):
    def test_usage_summary_is_complete(self):
        usage = load_usage()
        self.assertGreater(len(usage), 40)
        for name, entry in usage.items():
            self.assertTrue(entry["abilities"] and entry["items"] and entry["spreads"], name)
            self.assertGreaterEqual(len(entry["moves"]), 4, name)


@unittest.skipUnless(SHOWDOWN_BUILT, "run scripts/setup_showdown.sh first")
class GeneratePoolTest(unittest.TestCase):
    def test_teams_are_legal_distinct_and_repeatable(self):
        pool = generate_pool(4, seed=3)
        self.assertEqual(len(pool), 4)
        for team in pool:
            self.assertEqual(len(set(team["species"])), 6)
        self.assertEqual([t["species"] for t in pool], [t["species"] for t in generate_pool(4, seed=3)])


if __name__ == "__main__":
    unittest.main()
