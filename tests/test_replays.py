import json
import unittest
from pathlib import Path

from champions.import_replays import base_species, parse_replay

FIXTURE = Path(__file__).parent / "fixtures" / "replay_sample.json"


class ReplayParseTest(unittest.TestCase):
    def setUp(self):
        self.game = parse_replay(json.loads(FIXTURE.read_text()))

    def test_base_species_strips_details_and_mega(self):
        self.assertEqual(base_species("Raichu-Mega-Y, L50, F, shiny"), "Raichu")
        self.assertEqual(base_species("Meowstic-F-Mega, L50, F"), "Meowstic-F")
        self.assertEqual(base_species("Indeedee-F, L50, F"), "Indeedee-F")

    def test_winner_and_length(self):
        self.assertEqual(self.game["winner"], "p2")
        self.assertEqual(self.game["turns"], 7)

    def test_teams_leads_and_brought(self):
        p1 = self.game["p1"]
        self.assertEqual(len(p1["team"]), 6)
        self.assertEqual(p1["lead"], ["Raichu", "Milotic"])
        self.assertEqual(len(p1["brought"]), 4)

    def test_mega_uses_team_species_name(self):
        self.assertEqual(self.game["p1"]["mega"], ["Raichu"])
        self.assertEqual(self.game["p2"]["mega"], ["Meowstic-F"])

    def test_no_player_names_stored(self):
        self.assertNotIn("PlayerOne", json.dumps(self.game))


if __name__ == "__main__":
    unittest.main()


class ReplayCheckTest(unittest.TestCase):
    def test_auc_extremes_and_ties(self):
        from champions.replay_check import auc
        self.assertEqual(auc([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0]), 1.0)
        self.assertEqual(auc([0.1, 0.2, 0.8, 0.9], [1, 1, 0, 0]), 0.0)
        self.assertEqual(auc([0.5, 0.5, 0.5, 0.5], [1, 1, 0, 0]), 0.5)

    def test_usage_key_prefers_common_mega_form(self):
        from champions.replay_check import usage_key
        usage = {"Salamence": {"usage": 0.003}, "Salamence-Mega": {"usage": 0.28},
                 "Floette-Mega": {"usage": 0.1}, "Rillaboom": {"usage": 0.46}}
        self.assertEqual(usage_key("Salamence", False, usage), "Salamence-Mega")
        self.assertEqual(usage_key("Floette-Eternal", False, usage), "Floette-Mega")
        self.assertEqual(usage_key("Rillaboom", False, usage), "Rillaboom")
        self.assertIsNone(usage_key("Pangoro", False, usage))
