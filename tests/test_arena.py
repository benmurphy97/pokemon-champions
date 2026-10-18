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

    def test_a_plugged_in_team_preview_chooses_the_lineup(self):
        from champions.agent import NeuralAgent
        from champions.arena import paired_jobs, run_jobs
        from champions.policy import PolicyNet
        torch.manual_seed(0)
        pool = [{"text": (FIXTURES / name).read_text()} for name in ("team_tr_alpha.txt", "team_cybertron.txt")]
        asked = []

        def preview(state, side, request):
            asked.append(len(request["side"]["pokemon"]))
            return "team 5612"          # bring Pokémon 5, 6, 1 and 2; the first two listed lead

        agent = NeuralAgent(PolicyNet(dim=32, layers=1, heads=2), preview=preview)
        results = run_jobs(paired_jobs(pool, 4, "neural", "random", seed=2), agent, workers=1)
        self.assertEqual(len(results), 4)
        self.assertEqual(set(asked), {6})        # shown all six of its own team
        self.assertEqual(sum(r["fallbacks"] for r in results), 0)


if __name__ == "__main__":
    unittest.main()


@unittest.skipUnless(HAVE_TORCH, "needs torch (.venv)")
class ForcedMoveTest(unittest.TestCase):
    def test_locked_or_recharging_pokemon_just_plays_move_1(self):
        from champions.agent import NeuralAgent
        from champions.battle_state import BattleState
        from champions.policy import PolicyNet
        state = BattleState()
        for line in ["|poke|p1|Rillaboom, L50, M|", "|poke|p1|Incineroar, L50, M|", "|poke|p2|Sneasler, L50, M|",
                     "|switch|p1a: R|Rillaboom, L50, M|100/100", "|switch|p2a: S|Sneasler, L50, M|100/100", "|turn|2"]:
            state.update(line)
        request = {"active": [{"moves": [{"move": "Recharge", "id": "recharge"}]}],
                   "side": {"pokemon": [{"details": "Rillaboom, L50, M", "condition": "100/100", "active": True},
                                        {"details": "Incineroar, L50, M", "condition": "100/100", "active": False}]}}
        agent = NeuralAgent(PolicyNet(dim=32, layers=1, heads=2))
        self.assertEqual(agent.decide(state, "p1", request), "move 1")
