import unittest

import random

from champions.model.belief import Corpus, infer, sample_assignments, with_mega_items


def mon(name, item, moves, ability="A", nature="Jolly", evs=None):
    return {"name": name, "item": item, "ability": ability, "nature": nature, "evs": evs or {"HP": 2, "Atk": 32, "Def": 0, "SpA": 0, "SpD": 0, "Spe": 32}, "moves": moves}


def team(*mons):
    fillers = [mon(f"Filler{i}", f"Item{i}", ["Tackle"]) for i in range(6 - len(mons))]
    return list(mons) + fillers


CORPUS = Corpus([
    team(mon("Garchomp", "Life Orb", ["Earthquake", "Rock Slide", "Protect", "Dragon Claw"])),
    team(mon("Garchomp", "Life Orb", ["Earthquake", "Rock Slide", "Protect", "Dragon Claw"])),
    team(mon("Garchomp", "Choice Scarf", ["Earthquake", "Rock Slide", "Stomping Tantrum", "Dragon Claw"], nature="Adamant")),
])
SIX = ["Garchomp", "Filler0", "Filler1", "Filler2", "Filler3", "Filler4"]


def belief(observed):
    return infer(CORPUS, {}, SIX, [observed] + [{} for _ in range(5)])[0]


class BeliefTest(unittest.TestCase):
    def test_revealed_facts_are_certain(self):
        b = belief({"moves": ["Protect"], "item": "Life Orb"})
        self.assertEqual([m for m in b["moves"] if m["known"]], [{"name": "Protect", "p": 1.0, "known": True}])
        self.assertEqual(b["items"], [{"name": "Life Orb", "p": 1.0, "known": True}])

    def test_a_revealed_move_shifts_the_other_moves(self):
        base = {m["name"]: m["p"] for m in belief({"moves": []})["moves"]}
        after = {m["name"]: m["p"] for m in belief({"moves": ["Stomping Tantrum"]})["moves"]}
        self.assertLess(base["Stomping Tantrum"], 0.5)
        self.assertGreater(after.get("Dragon Claw", 0), 0.9)
        self.assertLess(after.get("Protect", 1), 0.1)      # only the Scarf set has Stomping Tantrum, and it has no Protect

    def test_four_known_moves_leave_nothing_else(self):
        b = belief({"moves": ["Earthquake", "Rock Slide", "Protect", "Dragon Claw"]})
        self.assertTrue(all(m["known"] for m in b["moves"]))

    def test_item_clause(self):
        out = infer(CORPUS, {}, SIX, [{}, {"item": "Life Orb"}] + [{} for _ in range(4)])
        self.assertEqual(out[0]["items"][0]["name"], "Choice Scarf")

    def test_samples_keep_revealed_facts_and_the_item_clause(self):
        observed = [{"moves": ["Stomping Tantrum"], "item": None, "ability": None}] + [{} for _ in range(5)]
        for assignment in sample_assignments(CORPUS, {}, SIX, observed, 20, random.Random(0)):
            first = assignment[0]
            self.assertIn("Stomping Tantrum", first["moves"])
            self.assertEqual(first["item"], "Choice Scarf")
            self.assertLessEqual(len(first["moves"]), 4)
            items = [m["item"] for m in assignment]
            self.assertEqual(len(items), len(set(items)))        # Item Clause

    def test_a_mega_form_reveals_its_stone(self):
        corpus = Corpus([team(mon("Garchomp-Mega-Z", "Garchompite Z", ["Earthquake", "Protect", "Dragon Claw", "Rock Slide"]))])
        out = with_mega_items(corpus, [{"form": "Garchomp-Mega-Z", "moves": []}])
        self.assertEqual(out[0]["item"], "Garchompite Z")


if __name__ == "__main__":
    unittest.main()
