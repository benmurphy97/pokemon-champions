import unittest
from pathlib import Path

from champions.simulate import RUNNER, run_battles, validate_team

FIXTURES = Path(__file__).parent / "fixtures"
SHOWDOWN_BUILT = (RUNNER.parent.parent / "vendor" / "pokemon-showdown" / "dist" / "sim").exists()


@unittest.skipUnless(SHOWDOWN_BUILT, "run scripts/setup_showdown.sh first")
class SimulateTest(unittest.TestCase):
    def setUp(self):
        self.a = (FIXTURES / "team_tr_alpha.txt").read_text()
        self.b = (FIXTURES / "team_cybertron.txt").read_text()

    def test_every_battle_finishes(self):
        result = run_battles(self.a, self.b, battles=20)
        self.assertEqual(result["winsA"] + result["winsB"] + result["ties"], 20)
        self.assertEqual(result["noResult"], 0)

    def test_same_seed_gives_same_result(self):
        first = run_battles(self.a, self.b, battles=20, seed=5)
        second = run_battles(self.a, self.b, battles=20, seed=5)
        self.assertEqual((first["winsA"], first["winsB"]), (second["winsA"], second["winsB"]))

    def test_legal_team_has_no_problems(self):
        self.assertEqual(validate_team(self.a), [])

    def test_overspent_stat_points_are_rejected(self):
        bad = self.a.replace("EVs: 32 HP / 32 SpA / 2 SpD", "EVs: 40 HP / 32 SpA / 20 SpD", 1)
        self.assertTrue(any("Stat Points" in p for p in validate_team(bad)))


if __name__ == "__main__":
    unittest.main()
