import random
from simulation.styles import INSIDE, MID, OUTSIDE

PRESS_FORWARD = "press_forward"
MAINTAIN_DISTANCE = "maintain_distance"
THROW_JAB = "throw_jab"
COUNTER_ATTACK = "counter_attack"
DEFENSIVE_SHELL = "defensive_shell"

ALL_ACTIONS = [PRESS_FORWARD, MAINTAIN_DISTANCE, THROW_JAB, COUNTER_ATTACK, DEFENSIVE_SHELL]

# Base action weights per preferred distance × current distance state.
# Encodes the core behavioral tendencies of each style.
_BASE_WEIGHTS: dict[str, dict[str, dict[str, float]]] = {
    # Philly Shell: always hunting inside range
    INSIDE: {
        OUTSIDE: {
            PRESS_FORWARD: 4.0,
            MAINTAIN_DISTANCE: 0.5,
            THROW_JAB: 0.5,
            COUNTER_ATTACK: 1.0,
            DEFENSIVE_SHELL: 0.5,
        },
        MID: {
            PRESS_FORWARD: 2.5,
            MAINTAIN_DISTANCE: 1.0,
            THROW_JAB: 0.8,
            COUNTER_ATTACK: 2.0,
            DEFENSIVE_SHELL: 1.2,
        },
        INSIDE: {
            PRESS_FORWARD: 0.5,
            MAINTAIN_DISTANCE: 1.0,
            THROW_JAB: 0.4,
            COUNTER_ATTACK: 3.5,
            DEFENSIVE_SHELL: 2.5,
        },
    },
    # Soviet Style: jab-and-hold, preserve outside range
    OUTSIDE: {
        OUTSIDE: {
            PRESS_FORWARD: 0.3,
            MAINTAIN_DISTANCE: 3.0,
            THROW_JAB: 3.5,
            COUNTER_ATTACK: 1.5,
            DEFENSIVE_SHELL: 0.7,
        },
        MID: {
            PRESS_FORWARD: 0.5,
            MAINTAIN_DISTANCE: 2.0,
            THROW_JAB: 2.5,
            COUNTER_ATTACK: 1.5,
            DEFENSIVE_SHELL: 1.0,
        },
        INSIDE: {
            PRESS_FORWARD: 0.2,
            MAINTAIN_DISTANCE: 3.0,
            THROW_JAB: 0.5,
            COUNTER_ATTACK: 1.0,
            DEFENSIVE_SHELL: 3.5,
        },
    },
}


def get_action(fighter, distance: str, positional_advantage: float, prev_outcome: str) -> str:
    """
    Rule-based action selection.

    positional_advantage: float in [-1.0, 1.0], positive = this fighter is ahead
    prev_outcome: 'won' | 'lost' | 'neutral'
    """
    style = fighter.style
    weights = dict(_BASE_WEIGHTS[style.preferred_distance][distance])

    # Positional advantage adjustments
    if positional_advantage > 0.3:
        weights[PRESS_FORWARD] *= 1.3
        weights[THROW_JAB] *= 1.2
        weights[DEFENSIVE_SHELL] *= 0.7
    elif positional_advantage < -0.3:
        weights[DEFENSIVE_SHELL] *= 1.5
        weights[COUNTER_ATTACK] *= 1.3
        weights[PRESS_FORWARD] *= 0.7

    # Momentum: reinforce or correct based on last exchange
    if prev_outcome == "won":
        weights[PRESS_FORWARD] *= 1.2
        weights[THROW_JAB] *= 1.1
    elif prev_outcome == "lost":
        weights[DEFENSIVE_SHELL] *= 1.3
        weights[COUNTER_ATTACK] *= 1.2
        weights[PRESS_FORWARD] *= 0.8

    # Style-specific trait amplifiers
    if style.counter_rate > 0.5:
        weights[COUNTER_ATTACK] *= 1 + style.counter_rate * 0.5
        weights[DEFENSIVE_SHELL] *= 1 + style.counter_rate * 0.3

    if style.jab_control > 0.6:
        weights[THROW_JAB] *= 1 + style.jab_control * 0.4
        weights[MAINTAIN_DISTANCE] *= 1 + style.jab_control * 0.2

    actions = list(weights.keys())
    w_values = [max(0.01, weights[a]) for a in actions]
    return random.choices(actions, weights=w_values, k=1)[0]
