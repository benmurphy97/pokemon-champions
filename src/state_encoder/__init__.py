"""
state_encoder — EncoderV2 for Pokémon Champions Regulation M-A.

Public API:
    from state_encoder import EncoderV2, STATE_DIM, GameState
    from state_encoder.game_state import (
        YourActivePokemon, YourBenchPokemon,
        OppActivePokemon, OppBenchSlot,
        FieldState, BattleContext,
        MoveState, VolatileStatus, PrimaryStatus,
        StatAlignment, ItemState, AbilityState,
        LastTurnSlot, ScreenState,
    )
"""

from .encoder_v2 import EncoderV2, STATE_DIM
from .game_state import (
    GameState,
    YourActivePokemon,
    YourBenchPokemon,
    OppActivePokemon,
    OppBenchSlot,
    FieldState,
    BattleContext,
    MoveState,
    VolatileStatus,
    PrimaryStatus,
    StatAlignment,
    ItemState,
    AbilityState,
    LastTurnSlot,
    ScreenState,
)
from .mega_table import MegaTable, MegaFormData

__all__ = [
    "EncoderV2",
    "STATE_DIM",
    "GameState",
    "YourActivePokemon",
    "YourBenchPokemon",
    "OppActivePokemon",
    "OppBenchSlot",
    "FieldState",
    "BattleContext",
    "MoveState",
    "VolatileStatus",
    "PrimaryStatus",
    "StatAlignment",
    "ItemState",
    "AbilityState",
    "LastTurnSlot",
    "ScreenState",
    "MegaTable",
    "MegaFormData",
]
