"""
Field state encoder for EncoderV2.

FIELD_DIM = 44 floats:
  weather_onehot(5) + weather_turns(1) + weather_permanent(1) + set_by_us(1)  = 8
  terrain_onehot(5) + terrain_turns(1)                                         = 6
  trick_room_active(1) + trick_room_turns(1)                                   = 2
  our_tailwind(1) + our_tw_turns(1) + opp_tailwind(1) + opp_tw_turns(1)       = 4
  our_light_screen(2) + our_reflect(2) + opp_light_screen(2) + opp_reflect(2) = 8
  gravity(2) + our_safeguard(2) + opp_safeguard(2)                            = 6
  magic_room(2) + wonder_room(2)                                               = 4
  TOTAL                                                                        = 38

Note: brief says ~40; exact count is 38.
"""
from __future__ import annotations

from .constants import (
    WEATHER_ORDER, N_WEATHER, TERRAIN_ORDER, N_TERRAIN,
    WEATHER_MAX, TERRAIN_MAX, TR_MAX, TAILWIND_MAX, SCREEN_MAX, TRAP_MAX,
)
from .game_state import FieldState, ScreenState

FIELD_DIM = 38


def _screen_vec(screen: ScreenState, max_turns: float = SCREEN_MAX) -> list[float]:
    return [
        float(screen.active),
        min(1.0, screen.turns_remaining / max_turns) if screen.active else 0.0,
    ]


def encode_field(field: FieldState) -> list[float]:
    """Return FIELD_DIM (38) floats describing the current field state."""
    v: list[float] = []

    # Weather (8)
    w_vec = [0.0] * N_WEATHER
    w_idx = WEATHER_ORDER.index(field.weather) if field.weather in WEATHER_ORDER else 0
    w_vec[w_idx] = 1.0
    v.extend(w_vec)
    v.append(min(1.0, field.weather_turns / WEATHER_MAX) if not field.weather_permanent else 1.0)
    v.append(float(field.weather_permanent))
    v.append(float(field.weather_set_by_our_side))

    # Terrain (6)
    t_vec = [0.0] * N_TERRAIN
    t_idx = TERRAIN_ORDER.index(field.terrain) if field.terrain in TERRAIN_ORDER else 0
    t_vec[t_idx] = 1.0
    v.extend(t_vec)
    v.append(min(1.0, field.terrain_turns / TERRAIN_MAX))

    # Trick Room (2)
    v.append(float(field.trick_room_active))
    v.append(min(1.0, field.trick_room_turns / TR_MAX) if field.trick_room_active else 0.0)

    # Tailwind (4)
    v.append(float(field.our_tailwind))
    v.append(min(1.0, field.our_tailwind_turns / TAILWIND_MAX) if field.our_tailwind else 0.0)
    v.append(float(field.opp_tailwind))
    v.append(min(1.0, field.opp_tailwind_turns / TAILWIND_MAX) if field.opp_tailwind else 0.0)

    # Screens (8)
    v.extend(_screen_vec(field.our_light_screen))
    v.extend(_screen_vec(field.our_reflect))
    v.extend(_screen_vec(field.opp_light_screen))
    v.extend(_screen_vec(field.opp_reflect))

    # Gravity (2)
    v.append(float(field.gravity_active))
    v.append(min(1.0, field.gravity_turns / 5.0) if field.gravity_active else 0.0)

    # Safeguard (4)
    v.extend(_screen_vec(field.our_safeguard, max_turns=5.0))
    v.extend(_screen_vec(field.opp_safeguard, max_turns=5.0))

    # Magic Room + Wonder Room (4)
    v.append(float(field.magic_room_active))
    v.append(min(1.0, field.magic_room_turns / 5.0) if field.magic_room_active else 0.0)
    v.append(float(field.wonder_room_active))
    v.append(min(1.0, field.wonder_room_turns / 5.0) if field.wonder_room_active else 0.0)

    assert len(v) == FIELD_DIM, f"field dim {len(v)} != {FIELD_DIM}"
    return v
