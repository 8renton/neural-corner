from ai.decision_policy import get_action, MAINTAIN_DISTANCE


class FighterAgent:
    """
    Stateful wrapper around a Fighter that maintains exchange-to-exchange
    momentum and feeds it into the decision policy.
    """

    def __init__(self, fighter):
        self.fighter = fighter
        self._last_outcome: str = "neutral"

    def choose_action(self, distance: str, positional_advantage: float) -> str:
        return get_action(self.fighter, distance, positional_advantage, self._last_outcome)

    def record_outcome(self, outcome: str) -> None:
        self._last_outcome = outcome

    def reset_momentum(self) -> None:
        """Called between rounds — clears single-exchange memory."""
        self._last_outcome = "neutral"

    def reset(self) -> None:
        """Full reset for a new fight."""
        self._last_outcome = "neutral"
