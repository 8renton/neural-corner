import random
from simulation.fighters import create_philly_shell_fighter, create_soviet_style_fighter
from simulation.engine import simulate_fight
from ai.agent import FighterAgent
from utils.stats import aggregate_fights, print_report

N_FIGHTS = 500
N_ROUNDS = 10
SEED = 42


def main() -> None:
    random.seed(SEED)

    fighter_a = create_philly_shell_fighter()
    fighter_b = create_soviet_style_fighter()

    print(f"\nNeural Corner — initializing")
    print(f"  {fighter_a}  vs  {fighter_b}")
    print(f"  {N_FIGHTS} fights × {N_ROUNDS} rounds  (seed={SEED})")

    agent_a = FighterAgent(fighter_a)
    agent_b = FighterAgent(fighter_b)

    results = []
    for _ in range(N_FIGHTS):
        result = simulate_fight(fighter_a, fighter_b, agent_a, agent_b, num_rounds=N_ROUNDS)
        results.append(result)

    stats = aggregate_fights(results, fighter_a.name, fighter_b.name)
    print_report(stats)


if __name__ == "__main__":
    main()
