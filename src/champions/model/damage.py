"""Damage arithmetic for the policy's inputs: how much of a target's health a move takes off.

The network is shown these numbers (it still learns what to do with them). They follow the game's
level-50 damage formula with the average damage roll, and include the things that swing a doubles
turn: weather, spread-move reduction, same-type bonus, burn, stat boosts, and the common held items
and abilities. Anything the model cannot see (a foe's unrevealed ability, a crit) is left out.

    calc = DamageCalc()
    hp_points = calc(attacker, defender, move_type, move_cat, move_power, move_target, two_targets, weather)

`attacker` and `defender` are dicts of tensors, one row per example: types [B,2], stats [B,6]
(hp, atk, def, spa, spd, spe), boosts [B,7] in stages, item [B], ability [B]; the attacker also has
burn [B] and statused [B]. Moves are given as [B,K] tensors of type/category/power/target ids.
"""

import torch
from torch import nn

from champions.model.vocab import ABILITY_ID, ITEM_ID, NUM_ABILITIES, NUM_ITEMS, TARGETS, TYPE_CHART, TYPES

LEVEL_TERM = 22.0   # 2 * level / 5 + 2, at level 50
ROLL = 0.925        # average of the 0.85-1.00 damage roll

TYPE_BOOST_ITEMS = {   # item -> the type it strengthens by 20%
    "Charcoal": "Fire", "Mystic Water": "Water", "Magnet": "Electric", "Miracle Seed": "Grass",
    "Never-Melt Ice": "Ice", "Black Belt": "Fighting", "Poison Barb": "Poison", "Soft Sand": "Ground",
    "Sharp Beak": "Flying", "Twisted Spoon": "Psychic", "Silver Powder": "Bug", "Hard Stone": "Rock",
    "Spell Tag": "Ghost", "Dragon Fang": "Dragon", "Black Glasses": "Dark", "Metal Coat": "Steel",
    "Silk Scarf": "Normal", "Fairy Feather": "Fairy",
}
ABILITY_IMMUNITIES = {   # ability -> the type of move it makes the holder immune to
    "Levitate": "Ground", "Earth Eater": "Ground", "Flash Fire": "Fire", "Well-Baked Body": "Fire",
    "Water Absorb": "Water", "Storm Drain": "Water", "Dry Skin": "Water", "Volt Absorb": "Electric",
    "Lightning Rod": "Electric", "Motor Drive": "Electric", "Sap Sipper": "Grass",
}
ABILITY_NORMAL_TO = {"Pixilate": "Fairy", "Aerilate": "Flying", "Refrigerate": "Ice", "Galvanize": "Electric"}   # Normal moves change type, +20%
ABILITY_AURAS = {"Fairy Aura": "Fairy", "Dark Aura": "Dark"}   # moves of this type get +33%


def type_id(name: str) -> int:
    return TYPES.index(name) + 1


def boost_multiplier(stage: torch.Tensor) -> torch.Tensor:
    return torch.where(stage >= 0, (2 + stage) / 2, 2 / (2 - stage))


def _table(size: int, default: float, ids: dict, dtype=torch.float32) -> torch.Tensor:
    """A lookup table indexed by item/ability id, `default` everywhere except the listed names."""
    table = torch.full((size,), default, dtype=dtype)
    for index, value in ids.items():
        table[index] = value
    return table


def _ids(names: dict, vocab: dict) -> dict:
    return {vocab[name]: value for name, value in names.items() if name in vocab}


class DamageCalc(nn.Module):
    def __init__(self):
        super().__init__()
        items = lambda names: _ids(names, ITEM_ID)
        abilities = lambda names: _ids(names, ABILITY_ID)
        buf = lambda name, tensor: self.register_buffer(name, tensor, persistent=False)

        buf("type_chart", torch.as_tensor(TYPE_CHART))
        buf("item_phys", _table(NUM_ITEMS, 1.0, items({"Life Orb": 1.3, "Muscle Band": 1.1})))
        buf("item_spec", _table(NUM_ITEMS, 1.0, items({"Life Orb": 1.3, "Wise Glasses": 1.1})))
        buf("item_expert", _table(NUM_ITEMS, 0.0, items({"Expert Belt": 1.0})))
        buf("item_type", _table(NUM_ITEMS, 0, items({k: type_id(v) for k, v in TYPE_BOOST_ITEMS.items()}), torch.long))
        buf("item_immune", _table(NUM_ITEMS, 0, items({"Air Balloon": type_id("Ground")}), torch.long))
        buf("ability_phys", _table(NUM_ABILITIES, 1.0, abilities({"Huge Power": 2.0, "Pure Power": 2.0})))
        buf("ability_guts", _table(NUM_ABILITIES, 0.0, abilities({"Guts": 1.0})))
        buf("ability_technician", _table(NUM_ABILITIES, 0.0, abilities({"Technician": 1.0})))
        buf("ability_adaptability", _table(NUM_ABILITIES, 0.0, abilities({"Adaptability": 1.0})))
        buf("ability_thick_fat", _table(NUM_ABILITIES, 0.0, abilities({"Thick Fat": 1.0})))
        buf("ability_normal_to", _table(NUM_ABILITIES, 0, abilities({k: type_id(v) for k, v in ABILITY_NORMAL_TO.items()}), torch.long))
        buf("ability_aura", _table(NUM_ABILITIES, 0, abilities({k: type_id(v) for k, v in ABILITY_AURAS.items()}), torch.long))
        buf("ability_immune", _table(NUM_ABILITIES, 0, abilities({k: type_id(v) for k, v in ABILITY_IMMUNITIES.items()}), torch.long))
        self.levitate, self.air_balloon = ABILITY_ID.get("Levitate", -1), ITEM_ID.get("Air Balloon", -1)
        self.spread_ids = (TARGETS.index("allAdjacentFoes") + 1, TARGETS.index("allAdjacent") + 1)   # ids are 1-based

    def forward(self, atk: dict, dfn: dict, move_type, move_cat, move_power, move_target,
                two_targets: torch.Tensor, weather: torch.Tensor, terrain: torch.Tensor) -> torch.Tensor:
        """Expected damage in HP points, [B, K]. `two_targets` [B]: spread moves lose a quarter when both foes are present.

        `weather` [B,4] and `terrain` [B,4] are one-hot (sun, rain, sand, snow; electric, grassy, misty, psychic).
        """
        col = lambda x: x.unsqueeze(-1)
        retyped = col(self.ability_normal_to[atk["ability"]])
        changed = (move_type == type_id("Normal")) & (retyped > 0)
        move_type = torch.where(changed, retyped.expand_as(move_type), move_type)
        special = move_cat == 2
        attacking = (move_cat <= 2) & (move_power > 0)
        sun, rain, sand, snow = (col(weather[:, i]) for i in range(4))

        a_boosts, d_boosts = atk["boosts"], dfn["boosts"]
        attack = torch.where(special, col(atk["stats"][:, 3] * boost_multiplier(a_boosts[:, 2])),
                             col(atk["stats"][:, 1] * boost_multiplier(a_boosts[:, 0]) * self.ability_phys[atk["ability"]]))
        defence = torch.where(special, col(dfn["stats"][:, 4] * boost_multiplier(d_boosts[:, 3])),
                              col(dfn["stats"][:, 2] * boost_multiplier(d_boosts[:, 1])))
        d_types = dfn["types"]
        rock, ice = (d_types == type_id("Rock")).any(-1), (d_types == type_id("Ice")).any(-1)
        defence = defence * (1 + 0.5 * (special * sand * col(rock) + ~special * snow * col(ice)))

        base = LEVEL_TERM * move_power * attack / defence.clamp(min=1.0) / 50 + 2

        eff = self.type_chart[move_type.unsqueeze(-1), d_types.unsqueeze(1)].prod(-1)             # [B, K]
        a_types = atk["types"]
        same_type = ((move_type == col(a_types[:, 0])) | (move_type == col(a_types[:, 1]))) & (move_type > 0)
        adapt = col(self.ability_adaptability[atk["ability"]])
        stab = torch.where(same_type, 1.5 + 0.5 * adapt, torch.ones_like(eff))
        fire, water = move_type == type_id("Fire"), move_type == type_id("Water")
        weather_mult = 1 + 0.5 * (sun * fire + rain * water) - 0.5 * (sun * water + rain * fire)
        guts = col(self.ability_guts[atk["ability"]]) * col(atk["statused"]) * ~special
        burn = torch.where(col(atk["burn"]) * ~special * (1 - guts) > 0, 0.5, 1.0)
        a_item = atk["item"]
        item = torch.where(special, col(self.item_spec[a_item]), col(self.item_phys[a_item]))
        item = item * (1 + 0.2 * (move_type == col(self.item_type[a_item]))) * (1 + 0.2 * col(self.item_expert[a_item]) * (eff > 1))
        aura = 1 + (1 / 3) * (move_type == col(self.ability_aura[atk["ability"]])) * (col(self.ability_aura[atk["ability"]]) > 0)
        ability = aura * (1 + 0.2 * changed) * (1 + 0.5 * guts) * (1 + 0.5 * col(self.ability_technician[atk["ability"]]) * (move_power <= 60))
        thick_fat = 1 - 0.5 * col(self.ability_thick_fat[dfn["ability"]]) * (fire | (move_type == type_id("Ice")))
        immune = (((move_type == col(self.ability_immune[dfn["ability"]])) & (col(self.ability_immune[dfn["ability"]]) > 0))
                  | ((move_type == col(self.item_immune[dfn["item"]])) & (col(self.item_immune[dfn["item"]]) > 0)))
        grounded = lambda t: col(((t["types"] != type_id("Flying")).all(-1) & (t["ability"] != self.levitate)
                                 & (t["item"] != self.air_balloon)).float())
        electric, grassy, misty, psychic = (col(terrain[:, i]) for i in range(4))
        terrain_boost = 1 + 0.3 * grounded(atk) * (electric * (move_type == type_id("Electric")) + grassy * (move_type == type_id("Grass"))
                                                   + psychic * (move_type == type_id("Psychic")))
        terrain_guard = 1 - 0.5 * grounded(dfn) * misty * (move_type == type_id("Dragon"))
        spread = (move_target == self.spread_ids[0]) | (move_target == self.spread_ids[1])
        spread_mult = 1 - 0.25 * spread * col(two_targets)

        damage = base * eff * stab * weather_mult * burn * item * ability * thick_fat * terrain_boost * terrain_guard * spread_mult * ROLL
        return damage * attacking * ~immune
