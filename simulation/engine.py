import random
from dataclasses import dataclass, field
from typing import List

from simulation.styles import INSIDE, MID, OUTSIDE, DISTANCE_STATES, DISTANCE_INDEX
from ai.decision_policy import (
    PRESS_FORWARD,
    MAINTAIN_DISTANCE,
    THROW_JAB,
    COUNTER_ATTACK,
    DEFENSIVE_SHELL,
)

# How committed each action is on offense and defense.
# Pressing hard = more offense, less defense. Shelling = opposite.
_OFFENSE_MOD = {
    PRESS_FORWARD:     1.15,
    MAINTAIN_DISTANCE: 0.90,
    THROW_JAB:         1.10,
    COUNTER_ATTACK:    0.75,  # base — counter bonus applied separately
    DEFENSIVE_SHELL:   0.50,
}

_DEFENSE_MOD = {
    PRESS_FORWARD:     0.80,  # exposed while moving in
    MAINTAIN_DISTANCE: 1.05,
    THROW_JAB:         0.90,
    COUNTER_ATTACK:    1.20,  # loaded to respond
    DEFENSIVE_SHELL:   1.35,
}

# Physical movement intent per action.
# Positive = closing distance (toward INSIDE), negative = opening (toward OUTSIDE).
# Style preferences are expressed through *action selection* in decision_policy,
# not through these intents — so MAINTAIN_DISTANCE is truly neutral.
_MOVEMENT_INTENT = {
    PRESS_FORWARD:      1.0,
    COUNTER_ATTACK:     0.2,
    MAINTAIN_DISTANCE:  0.0,
    THROW_JAB:         -0.5,  # jab pushes opponent back
    DEFENSIVE_SHELL:   -0.1,
}

# Actions that leave the attacker open to a well-timed counter
_COMMITTED_ACTIONS = {PRESS_FORWARD, THROW_JAB}


@dataclass
class ExchangeResult:
    winner: str           # 'a' | 'b' | 'neutral'
    punches_a: float
    punches_b: float
    new_distance: str
    positional_advantage: float  # updated value, relative to fighter_a
    action_a: str
    action_b: str


@dataclass
class RoundResult:
    round_num: int
    winner: str           # 'a' | 'b' | 'draw'
    punches_a: float
    punches_b: float
    exchanges: int
    distance_counts: dict  # {distance: int}
    final_positional_advantage: float
    exchange_log: list = field(default_factory=list)  # List[ExchangeResult]


@dataclass
class FightResult:
    winner: str
    rounds_a: int
    rounds_b: int
    draw_rounds: int
    total_rounds: int
    rounds: List[RoundResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _height_offense_mod(fighter, opponent, distance: str) -> float:
    """
    Small reach-based modifier to offense.
    Longer reach helps at range; hurts in a phone booth.
    """
    reach = fighter.reach_advantage(opponent)  # positive if taller
    if distance == OUTSIDE:
        return reach * 0.06
    elif distance == MID:
        return reach * 0.03
    else:  # INSIDE: longer arms are cramped
        return reach * -0.04


def _resolve_distance(action_a, action_b, current: str, fighter_a, fighter_b) -> str:
    """
    Probabilistic distance shift.

    Net movement is the sum of each fighter's movement intent scaled by their
    relevant movement ability.  Positive net → closer (toward INSIDE).
    """
    idx = DISTANCE_INDEX[current]

    intent_a = _MOVEMENT_INTENT[action_a]
    intent_b = _MOVEMENT_INTENT[action_b]

    # Use the appropriate movement stat depending on intended direction
    ability_a = fighter_a.style.clinch_ability if intent_a >= 0 else fighter_a.style.distance_creation
    ability_b = fighter_b.style.clinch_ability if intent_b >= 0 else fighter_b.style.distance_creation

    net = (intent_a * ability_a) + (intent_b * ability_b)

    threshold = 0.12
    if abs(net) < threshold:
        return current

    shift_prob = min(0.70, abs(net) * 0.85)
    if random.random() >= shift_prob:
        return current

    new_idx = max(0, idx - 1) if net > 0 else min(2, idx + 1)
    return DISTANCE_STATES[new_idx]


# ---------------------------------------------------------------------------
# Simulation layers
# ---------------------------------------------------------------------------

def resolve_exchange_from_actions(
    fighter_a,
    fighter_b,
    action_a: str,
    action_b: str,
    distance: str,
    positional_advantage: float,
) -> ExchangeResult:
    """
    Pure exchange physics given explicit action strings.
    No agent callbacks — used directly by the RL training environment.
    """
    h_mod_a = _height_offense_mod(fighter_a, fighter_b, distance)
    h_mod_b = _height_offense_mod(fighter_b, fighter_a, distance)

    off_a = fighter_a.style.offense[distance] * _OFFENSE_MOD[action_a] + h_mod_a
    def_b = fighter_b.style.defense[distance] * _DEFENSE_MOD[action_b]
    off_b = fighter_b.style.offense[distance] * _OFFENSE_MOD[action_b] + h_mod_b
    def_a = fighter_a.style.defense[distance] * _DEFENSE_MOD[action_a]

    if action_a == COUNTER_ATTACK and action_b in _COMMITTED_ACTIONS:
        off_a += fighter_a.style.counter_rate * 0.30
        def_b *= 0.85
    if action_b == COUNTER_ATTACK and action_a in _COMMITTED_ACTIONS:
        off_b += fighter_b.style.counter_rate * 0.30
        def_a *= 0.85

    if positional_advantage > 0:
        off_a *= 1 + positional_advantage * 0.12
        def_a *= 1 + positional_advantage * 0.08
    else:
        mag = abs(positional_advantage)
        off_b *= 1 + mag * 0.12
        def_b *= 1 + mag * 0.08

    off_a = max(0.0, min(1.0, off_a))
    off_b = max(0.0, min(1.0, off_b))
    def_a = max(0.0, min(1.0, def_a))
    def_b = max(0.0, min(1.0, def_b))

    p_a = off_a * (1.0 - def_b)
    p_b = off_b * (1.0 - def_a)

    punches_a = 1.0 if random.random() < p_a else 0.0
    punches_b = 1.0 if random.random() < p_b else 0.0

    if punches_a > punches_b:
        winner, delta = "a", 0.08
    elif punches_b > punches_a:
        winner, delta = "b", -0.08
    else:
        winner, delta = "neutral", 0.0

    if distance == fighter_a.style.preferred_distance:
        delta += 0.02
    if distance == fighter_b.style.preferred_distance:
        delta -= 0.02

    return ExchangeResult(
        winner=winner,
        punches_a=punches_a,
        punches_b=punches_b,
        new_distance=_resolve_distance(action_a, action_b, distance, fighter_a, fighter_b),
        positional_advantage=max(-1.0, min(1.0, positional_advantage + delta)),
        action_a=action_a,
        action_b=action_b,
    )


def simulate_exchange(
    fighter_a,
    fighter_b,
    agent_a,
    agent_b,
    distance: str,
    positional_advantage: float,
) -> ExchangeResult:
    """Atomic unit: agents choose actions, physics resolves the exchange."""
    action_a = agent_a.choose_action(distance, positional_advantage)
    action_b = agent_b.choose_action(distance, -positional_advantage)

    result = resolve_exchange_from_actions(
        fighter_a, fighter_b, action_a, action_b, distance, positional_advantage
    )

    agent_a.record_outcome("won" if result.winner == "a" else "lost" if result.winner == "b" else "neutral")
    agent_b.record_outcome("won" if result.winner == "b" else "lost" if result.winner == "a" else "neutral")

    return result


def simulate_round(fighter_a, fighter_b, agent_a, agent_b, round_num: int) -> RoundResult:
    """Simulate one round (8–12 exchanges). Rounds start at mid range."""

    agent_a.reset_momentum()
    agent_b.reset_momentum()

    distance = MID
    positional_advantage = 0.0
    num_exchanges = random.randint(8, 12)

    total_a = 0.0
    total_b = 0.0
    distance_counts = {INSIDE: 0, MID: 0, OUTSIDE: 0}
    exchange_log = []

    for _ in range(num_exchanges):
        result = simulate_exchange(
            fighter_a, fighter_b, agent_a, agent_b, distance, positional_advantage
        )
        total_a += result.punches_a
        total_b += result.punches_b
        distance_counts[distance] += 1
        exchange_log.append(result)
        distance = result.new_distance
        positional_advantage = result.positional_advantage

    margin = total_a - total_b
    if abs(margin) <= 1.0:
        round_winner = "draw"
    elif margin > 0:
        round_winner = "a"
    else:
        round_winner = "b"

    return RoundResult(
        round_num=round_num,
        winner=round_winner,
        punches_a=total_a,
        punches_b=total_b,
        exchanges=num_exchanges,
        distance_counts=distance_counts,
        final_positional_advantage=positional_advantage,
        exchange_log=exchange_log,
    )


def simulate_fight(
    fighter_a, fighter_b, agent_a, agent_b, num_rounds: int = 10
) -> FightResult:
    """Simulate a complete fight and return the scored result."""

    agent_a.reset()
    agent_b.reset()

    rounds: List[RoundResult] = []
    rounds_a = rounds_b = draw_rounds = 0

    for r in range(1, num_rounds + 1):
        rnd = simulate_round(fighter_a, fighter_b, agent_a, agent_b, r)
        rounds.append(rnd)
        if rnd.winner == "a":
            rounds_a += 1
        elif rnd.winner == "b":
            rounds_b += 1
        else:
            draw_rounds += 1

    if rounds_a > rounds_b:
        winner = "a"
    elif rounds_b > rounds_a:
        winner = "b"
    else:
        winner = "draw"

    return FightResult(
        winner=winner,
        rounds_a=rounds_a,
        rounds_b=rounds_b,
        draw_rounds=draw_rounds,
        total_rounds=num_rounds,
        rounds=rounds,
    )
