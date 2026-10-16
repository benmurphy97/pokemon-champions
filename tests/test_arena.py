import json
import unittest
from pathlib import Path

try:
    import torch
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False

from champions.simulate import RUNNER

FIXTURES = Path(__file__).parent / "fixtures"
SHOWDOWN_BUILT = (RUNNER.parent.parent / "vendor" / "pokemon-showdown" / "dist" / "sim").exists()


@unittest.skipUnless(HAVE_TORCH and SHOWDOWN_BUILT, "needs torch (.venv) and the built Showdown engine")
class ArenaTest(unittest.TestCase):
    def test_untrained_policy_finishes_games_with_legal_choices(self):
        from champions.agent import NeuralAgent
        from champions.arena import challenger_wins, paired_jobs, run_jobs
        from champions.policy import PolicyNet
        torch.manual_seed(0)
        pool = [{"text": (FIXTURES / name).read_text()} for name in ("team_tr_alpha.txt", "team_cybertron.txt")]
        agent = NeuralAgent(PolicyNet(dim=32, layers=1, heads=2))
        results = run_jobs(paired_jobs(pool, 8, "neural", "random", seed=1), agent, workers=2)
        wins, decided, fallbacks = challenger_wins(results)
        self.assertEqual(len(results), 8)
        self.assertEqual(decided, 8)
        self.assertEqual(fallbacks, 0)   # every choice the policy made was accepted by the engine


if __name__ == "__main__":
    unittest.main()
