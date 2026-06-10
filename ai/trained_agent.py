"""
Agent wrapper for a trained PPO model.
Implements the same interface as FighterAgent so it can be dropped in
anywhere the engine expects an agent.
"""
import numpy as np
from stable_baselines3 import PPO

from ai.decision_policy import ALL_ACTIONS
from simulation.styles import DISTANCE_INDEX


OBS_SIZE = 13
NUM_ACTIONS = len(ALL_ACTIONS)


class TrainedAgent:
    def __init__(self, fighter, model_path: str):
        self.fighter = fighter
        self.model = PPO.load(model_path)

        # Obs state — updated each call to choose_action
        self._last_action_idx: int = 0
        self._pos_adv: float = 0.0
        self._rounds_self: int = 0
        self._rounds_opp: int = 0
        self._exchange_num: int = 0
        self._max_exchanges: int = 10
        self._round_num: int = 1
        self._num_rounds: int = 10

    # ------------------------------------------------------------------
    # Engine-compatible interface
    # ------------------------------------------------------------------

    def choose_action(self, distance: str, positional_advantage: float) -> str:
        self._pos_adv = positional_advantage
        obs = self._build_obs(distance, positional_advantage)
        action_idx, _ = self.model.predict(obs, deterministic=False)
        self._last_action_idx = int(action_idx)
        return ALL_ACTIONS[self._last_action_idx]

    def record_outcome(self, outcome: str) -> None:
        pass  # not used in obs — model relies on positional advantage instead

    def reset_momentum(self) -> None:
        self._exchange_num = 0

    def reset(self) -> None:
        self._last_action_idx = 0
        self._pos_adv = 0.0
        self._rounds_self = 0
        self._rounds_opp = 0
        self._exchange_num = 0
        self._round_num = 1

    # ------------------------------------------------------------------
    # Optional: called by the fight viewer to keep progress obs accurate
    # ------------------------------------------------------------------

    def update_progress(
        self,
        round_num: int,
        exchange_num: int,
        max_exchanges: int,
        rounds_self: int,
        rounds_opp: int,
        num_rounds: int = 10,
    ) -> None:
        self._round_num = round_num
        self._exchange_num = exchange_num
        self._max_exchanges = max_exchanges
        self._rounds_self = rounds_self
        self._rounds_opp = rounds_opp
        self._num_rounds = num_rounds

    # ------------------------------------------------------------------

    def _build_obs(self, distance: str, pos_adv: float) -> np.ndarray:
        dist_oh = np.zeros(3, dtype=np.float32)
        dist_oh[DISTANCE_INDEX[distance]] = 1.0

        action_oh = np.zeros(NUM_ACTIONS, dtype=np.float32)
        action_oh[self._last_action_idx] = 1.0

        exch_prog = self._exchange_num / max(1, self._max_exchanges)
        round_prog = (self._round_num - 1) / max(1, self._num_rounds)

        return np.concatenate([
            dist_oh,
            [pos_adv],
            action_oh,
            [self._rounds_self / max(1, self._num_rounds)],
            [self._rounds_opp / max(1, self._num_rounds)],
            [exch_prog],
            [round_prog],
        ]).astype(np.float32)
