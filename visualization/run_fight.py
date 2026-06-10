"""
Launch a live Pygame fight simulation.

Usage:
  python visualization/run_fight.py            # rule-based vs rule-based
  python visualization/run_fight.py --trained  # use trained models if available
  python visualization/run_fight.py --seed 7   # reproducible fight
"""
import sys
import os
import random
import argparse
import pygame

# Allow running from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from simulation.fighters import create_philly_shell_fighter, create_soviet_style_fighter
from simulation.engine import simulate_fight
from ai.agent import FighterAgent
from visualization.fight_viewer import FightViewer

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "training", "trained_models")
MODEL_A    = os.path.join(MODELS_DIR, "fighter_a_final.zip")
MODEL_B    = os.path.join(MODELS_DIR, "fighter_b_final.zip")


def build_agents(fighter_a, fighter_b, use_trained: bool):
    """Return (agent_a, agent_b, mode_label)."""
    if use_trained:
        from ai.trained_agent import TrainedAgent
        has_a = os.path.exists(MODEL_A)
        has_b = os.path.exists(MODEL_B)

        if has_a and has_b:
            print("  Loading trained models for both fighters.")
            return (
                TrainedAgent(fighter_a, MODEL_A.replace(".zip", "")),
                TrainedAgent(fighter_b, MODEL_B.replace(".zip", "")),
                "Trained  vs  Trained",
            )
        elif has_a:
            print("  Fighter A trained — Fighter B using rule-based policy.")
            return (
                TrainedAgent(fighter_a, MODEL_A.replace(".zip", "")),
                FighterAgent(fighter_b),
                "Trained  vs  Rule-based",
            )
        elif has_b:
            print("  Fighter B trained — Fighter A using rule-based policy.")
            return (
                FighterAgent(fighter_a),
                TrainedAgent(fighter_b, MODEL_B.replace(".zip", "")),
                "Rule-based  vs  Trained",
            )
        else:
            print("  No trained models found — falling back to rule-based.")
            print(f"  Train first:  python -m training.train")

    return (FighterAgent(fighter_a), FighterAgent(fighter_b), "Rule-based  vs  Rule-based")


def main():
    parser = argparse.ArgumentParser(description="Neural Corner fight viewer")
    parser.add_argument("--trained", action="store_true", help="Use trained models")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--rounds", type=int, default=10, help="Number of rounds")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    fighter_a = create_philly_shell_fighter()
    fighter_b = create_soviet_style_fighter()

    agent_a, agent_b, mode_label = build_agents(fighter_a, fighter_b, args.trained)

    print(f"\nNeural Corner — Fight Viewer")
    print(f"  {fighter_a}  vs  {fighter_b}")
    print(f"  Mode: {mode_label}")
    print(f"  Rounds: {args.rounds}")
    print("  Simulating fight...")

    fight_result = simulate_fight(fighter_a, fighter_b, agent_a, agent_b, num_rounds=args.rounds)

    print(f"  Result: {fighter_a.name if fight_result.winner == 'a' else fighter_b.name if fight_result.winner == 'b' else 'Draw'} wins")
    print("  Launching viewer (ESC to quit, SPACE to skip scenes)\n")

    viewer = FightViewer(fight_result, fighter_a, fighter_b)
    viewer.run()

    pygame.quit()


if __name__ == "__main__":
    main()
