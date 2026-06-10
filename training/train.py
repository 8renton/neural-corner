"""
Curriculum training for Neural Corner fighters.

Phase 1 — Style Drills
  Each fighter trains against a scripted sparring partner that emphasizes
  the opponent's core tendencies (range control vs pressure).

Phase 2 — Film Study
  Each fighter now faces the other's Phase 1 trained policy.

Phase 3 — Self-Play  (3 alternating iterations)
  Fighters train against each other's latest policy, adapting in response.

Usage:
  python -m training.train
"""
import os
import sys
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from training.boxing_env import BoxingEnv, ACTION_INDEX
from simulation.fighters import create_philly_shell_fighter, create_soviet_style_fighter
from simulation.styles import DISTANCE_STATES, DISTANCE_INDEX
from ai.decision_policy import (
    ALL_ACTIONS, get_action,
    PRESS_FORWARD, THROW_JAB, COUNTER_ATTACK, DEFENSIVE_SHELL, MAINTAIN_DISTANCE,
)

MODELS_DIR = os.path.join(os.path.dirname(__file__), "trained_models")
os.makedirs(MODELS_DIR, exist_ok=True)

fighter_a = create_philly_shell_fighter()
fighter_b = create_soviet_style_fighter()

PPO_KWARGS = dict(
    policy="MlpPolicy",
    verbose=0,
    learning_rate=3e-4,
    n_steps=512,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
)


# ---------------------------------------------------------------------------
# Opponent policy types
# ---------------------------------------------------------------------------

class RuleBasedOpponent:
    """Wraps the existing probabilistic rule-based agent."""

    def __init__(self, fighter):
        self.fighter = fighter
        self._last_outcome = "neutral"

    def __call__(self, obs: np.ndarray) -> int:
        dist_idx = int(np.argmax(obs[:3]))
        distance = DISTANCE_STATES[dist_idx]
        pos_adv = float(obs[3])
        action = get_action(self.fighter, distance, pos_adv, self._last_outcome)
        return ACTION_INDEX[action]


class DrillOpponent:
    """
    Phase 1 scripted sparring partner.

    'pressure'      — mimics a pure pressure/clinch fighter.
    'range_control' — mimics a pure range-management/jab fighter.
    """

    def __init__(self, strategy: str):
        assert strategy in ("pressure", "range_control")
        self.strategy = strategy

    def __call__(self, obs: np.ndarray) -> int:
        dist_idx = int(np.argmax(obs[:3]))
        distance = DISTANCE_STATES[dist_idx]

        if self.strategy == "pressure":
            if distance == "inside":
                return ACTION_INDEX[COUNTER_ATTACK]
            return ACTION_INDEX[PRESS_FORWARD]

        else:  # range_control
            if distance == "outside":
                return ACTION_INDEX[THROW_JAB]
            if distance == "mid":
                return ACTION_INDEX[THROW_JAB] if np.random.random() > 0.4 else ACTION_INDEX[MAINTAIN_DISTANCE]
            return ACTION_INDEX[DEFENSIVE_SHELL]


def model_as_policy(model: PPO):
    """Return a callable opponent policy backed by a trained SB3 model."""
    def policy(obs: np.ndarray) -> int:
        action, _ = model.predict(obs, deterministic=False)
        return int(action)
    return policy


# ---------------------------------------------------------------------------
# Training helper
# ---------------------------------------------------------------------------

def train_phase(
    fighter_self,
    fighter_opp,
    opponent_policy,
    phase_name: str,
    total_timesteps: int,
    existing_model: PPO = None,
) -> PPO:
    def make_env():
        return BoxingEnv(fighter_self, fighter_opp, opponent_policy)

    env = make_vec_env(make_env, n_envs=4)

    if existing_model is not None:
        existing_model.set_env(env)
        model = existing_model
    else:
        model = PPO(env=env, **PPO_KWARGS)

    model.learn(total_timesteps=total_timesteps, reset_num_timesteps=(existing_model is None))
    save_path = os.path.join(MODELS_DIR, phase_name)
    model.save(save_path)
    print(f"    saved → {phase_name}.zip")
    env.close()
    return model


# ---------------------------------------------------------------------------
# Curriculum
# ---------------------------------------------------------------------------

def run_curriculum():
    print("\nNeural Corner — Curriculum Training")
    print("=" * 50)

    # ── Phase 1: Style Drills ─────────────────────────────
    print("\n[Phase 1]  Style Drills")
    print(f"  {fighter_a.name}: training against range-control drill opponent")
    model_a = train_phase(
        fighter_a, fighter_b,
        DrillOpponent("range_control"),
        "fighter_a_phase1",
        total_timesteps=80_000,
    )

    print(f"  {fighter_b.name}: training against pressure drill opponent")
    model_b = train_phase(
        fighter_b, fighter_a,
        DrillOpponent("pressure"),
        "fighter_b_phase1",
        total_timesteps=80_000,
    )

    # ── Phase 2: Film Study ───────────────────────────────
    print("\n[Phase 2]  Film Study  (face the opponent's Phase 1 policy)")
    print(f"  {fighter_a.name}: studying Phase 1 {fighter_b.name}")
    model_a = train_phase(
        fighter_a, fighter_b,
        model_as_policy(model_b),
        "fighter_a_phase2",
        total_timesteps=100_000,
    )

    print(f"  {fighter_b.name}: studying Phase 1 {fighter_a.name}")
    model_b = train_phase(
        fighter_b, fighter_a,
        model_as_policy(model_a),
        "fighter_b_phase2",
        total_timesteps=100_000,
    )

    # ── Phase 3: Self-Play ────────────────────────────────
    print("\n[Phase 3]  Self-Play  (3 alternating iterations)")
    for i in range(3):
        print(f"  Iteration {i + 1}/3")
        model_a = train_phase(
            fighter_a, fighter_b,
            model_as_policy(model_b),
            f"fighter_a_selfplay_{i}",
            total_timesteps=80_000,
        )
        model_b = train_phase(
            fighter_b, fighter_a,
            model_as_policy(model_a),
            f"fighter_b_selfplay_{i}",
            total_timesteps=80_000,
        )

    # Save final models with clean names
    model_a.save(os.path.join(MODELS_DIR, "fighter_a_final"))
    model_b.save(os.path.join(MODELS_DIR, "fighter_b_final"))

    print("\n✓ Training complete.")
    print(f"  Final models in: {MODELS_DIR}/")
    print("  Watch a fight:  python visualization/run_fight.py --trained")


if __name__ == "__main__":
    run_curriculum()
