"""
Gymnasium environment wrapping Neural Corner's exchange model.

One episode = one full fight (num_rounds rounds).
The agent controls Fighter A; the opponent uses a supplied policy callable.

Observation (13 floats):
  [dist_oh(3), pos_adv(1), last_action_oh(5), rounds_self(1), rounds_opp(1),
   exchange_progress(1), round_progress(1)]

Action: Discrete(5) → index into ALL_ACTIONS
"""
import random
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from simulation.styles import INSIDE, MID, OUTSIDE, DISTANCE_INDEX, DISTANCE_STATES
from simulation.engine import resolve_exchange_from_actions
from ai.decision_policy import ALL_ACTIONS

ACTION_INDEX = {a: i for i, a in enumerate(ALL_ACTIONS)}
NUM_ACTIONS = len(ALL_ACTIONS)
OBS_SIZE = 13  # 3 + 1 + 5 + 1 + 1 + 1 + 1


class BoxingEnv(gym.Env):
    metadata = {}

    def __init__(self, fighter_self, fighter_opp, opponent_policy, num_rounds: int = 10):
        """
        fighter_self: the fighter this agent controls (always from its own perspective)
        fighter_opp:  the opponent fighter
        opponent_policy: callable(obs: np.ndarray) -> int  (action index)
        """
        super().__init__()
        self.fighter_self = fighter_self
        self.fighter_opp = fighter_opp
        self.opponent_policy = opponent_policy
        self.num_rounds = num_rounds

        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(OBS_SIZE,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(NUM_ACTIONS)

        self._reset_state()

    def _reset_state(self):
        self._distance = MID
        self._pos_adv = 0.0         # positive = self is ahead
        self._round_num = 1
        self._exchange_num = 0
        self._max_exchanges = 10
        self._rounds_self = 0
        self._rounds_opp = 0
        self._round_punches_self = 0.0
        self._round_punches_opp = 0.0
        self._last_action = 0       # index into ALL_ACTIONS

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_state()
        self._max_exchanges = random.randint(8, 12)
        return self._get_obs(), {}

    def step(self, action_idx: int):
        action_self = ALL_ACTIONS[int(action_idx)]

        opp_obs = self._get_opp_obs()
        action_opp = ALL_ACTIONS[int(self.opponent_policy(opp_obs))]

        result = resolve_exchange_from_actions(
            self.fighter_self, self.fighter_opp,
            action_self, action_opp,
            self._distance, self._pos_adv,
        )

        self._last_action = int(action_idx)
        self._distance = result.new_distance
        self._pos_adv = result.positional_advantage
        self._round_punches_self += result.punches_a
        self._round_punches_opp += result.punches_b
        self._exchange_num += 1

        # Dense per-exchange reward
        reward = (result.punches_a - result.punches_b) * 0.3

        terminated = False

        if self._exchange_num >= self._max_exchanges:
            margin = self._round_punches_self - self._round_punches_opp
            if margin > 1.0:
                self._rounds_self += 1
                reward += 1.0
            elif margin < -1.0:
                self._rounds_opp += 1
                reward -= 1.0

            self._round_num += 1
            self._exchange_num = 0
            self._max_exchanges = random.randint(8, 12)
            self._round_punches_self = 0.0
            self._round_punches_opp = 0.0
            self._distance = MID

            if self._round_num > self.num_rounds:
                terminated = True
                if self._rounds_self > self._rounds_opp:
                    reward += 3.0
                elif self._rounds_opp > self._rounds_self:
                    reward -= 3.0
                else:
                    reward += 0.5

        return self._get_obs(), float(reward), terminated, False, {}

    def _get_obs(self) -> np.ndarray:
        dist_oh = np.zeros(3, dtype=np.float32)
        dist_oh[DISTANCE_INDEX[self._distance]] = 1.0

        action_oh = np.zeros(NUM_ACTIONS, dtype=np.float32)
        action_oh[self._last_action] = 1.0

        exch_prog = self._exchange_num / max(1, self._max_exchanges)
        round_prog = (self._round_num - 1) / self.num_rounds

        return np.concatenate([
            dist_oh,
            [self._pos_adv],
            action_oh,
            [self._rounds_self / self.num_rounds],
            [self._rounds_opp / self.num_rounds],
            [exch_prog],
            [round_prog],
        ]).astype(np.float32)

    def _get_opp_obs(self) -> np.ndarray:
        """Build observation from the opponent's perspective (flipped pos_adv)."""
        dist_oh = np.zeros(3, dtype=np.float32)
        dist_oh[DISTANCE_INDEX[self._distance]] = 1.0

        # Opponent's last action is unknown to self; use zeros for opponent's self-obs
        action_oh = np.zeros(NUM_ACTIONS, dtype=np.float32)

        exch_prog = self._exchange_num / max(1, self._max_exchanges)
        round_prog = (self._round_num - 1) / self.num_rounds

        return np.concatenate([
            dist_oh,
            [-self._pos_adv],
            action_oh,
            [self._rounds_opp / self.num_rounds],
            [self._rounds_self / self.num_rounds],
            [exch_prog],
            [round_prog],
        ]).astype(np.float32)

    def render(self):
        pass
