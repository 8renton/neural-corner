from simulation.engine import FightResult, RoundResult
from simulation.styles import INSIDE, MID, OUTSIDE


def aggregate_fights(
    results: list,
    fighter_a_name: str,
    fighter_b_name: str,
) -> dict:
    """Compute aggregate statistics across all simulated fights."""
    n = len(results)

    wins_a = sum(1 for r in results if r.winner == "a")
    wins_b = sum(1 for r in results if r.winner == "b")
    draws = sum(1 for r in results if r.winner == "draw")

    avg_rounds_a = sum(r.rounds_a for r in results) / n
    avg_rounds_b = sum(r.rounds_b for r in results) / n
    avg_draw_rounds = sum(r.draw_rounds for r in results) / n

    all_rounds: list[RoundResult] = []
    for fight in results:
        all_rounds.extend(fight.rounds)

    early_wins_a = early_wins_b = 0
    late_wins_a = late_wins_b = 0

    for rnd in all_rounds:
        if rnd.round_num <= 4:
            if rnd.winner == "a":
                early_wins_a += 1
            elif rnd.winner == "b":
                early_wins_b += 1
        else:
            if rnd.winner == "a":
                late_wins_a += 1
            elif rnd.winner == "b":
                late_wins_b += 1

    total_early = early_wins_a + early_wins_b
    total_late = late_wins_a + late_wins_b

    dist_counts = {INSIDE: 0, MID: 0, OUTSIDE: 0}
    for rnd in all_rounds:
        for d, c in rnd.distance_counts.items():
            dist_counts[d] += c

    total_dist = sum(dist_counts.values()) or 1

    avg_punches_a = sum(rnd.punches_a for rnd in all_rounds) / len(all_rounds)
    avg_punches_b = sum(rnd.punches_b for rnd in all_rounds) / len(all_rounds)

    return {
        "n_fights": n,
        "wins_a": wins_a,
        "wins_b": wins_b,
        "draws": draws,
        "win_pct_a": wins_a / n * 100,
        "win_pct_b": wins_b / n * 100,
        "draw_pct": draws / n * 100,
        "avg_rounds_won_a": avg_rounds_a,
        "avg_rounds_won_b": avg_rounds_b,
        "avg_draw_rounds": avg_draw_rounds,
        "early_win_pct_a": (early_wins_a / total_early * 100) if total_early else 0.0,
        "early_win_pct_b": (early_wins_b / total_early * 100) if total_early else 0.0,
        "late_win_pct_a": (late_wins_a / total_late * 100) if total_late else 0.0,
        "late_win_pct_b": (late_wins_b / total_late * 100) if total_late else 0.0,
        "distance_pct": {d: dist_counts[d] / total_dist * 100 for d in [INSIDE, MID, OUTSIDE]},
        "avg_punches_per_round_a": avg_punches_a,
        "avg_punches_per_round_b": avg_punches_b,
        "fighter_a_name": fighter_a_name,
        "fighter_b_name": fighter_b_name,
    }


def print_report(stats: dict) -> None:
    a = stats["fighter_a_name"]
    b = stats["fighter_b_name"]
    n = stats["n_fights"]
    dp = stats["distance_pct"]

    sep = "=" * 62

    print(f"\n{sep}")
    print("  NEURAL CORNER  ·  BOXING SIMULATION ANALYTICS REPORT")
    print(sep)
    print(f"  Fights simulated : {n:,}")
    print(f"  {a}  vs  {b}")
    print(f"  {'─' * 58}")

    print("\n  FIGHT OUTCOMES")
    print(f"  {'Fighter':<28}  {'Win %':>6}  {'Fights':>7}")
    print(f"  {'─' * 44}")
    print(f"  {a:<28}  {stats['win_pct_a']:>5.1f}%  {stats['wins_a']:>7,}")
    print(f"  {b:<28}  {stats['win_pct_b']:>5.1f}%  {stats['wins_b']:>7,}")
    print(f"  {'Draws':<28}  {stats['draw_pct']:>5.1f}%  {stats['draws']:>7,}")

    print("\n  ROUND CONTROL  (avg rounds won per fight)")
    print(f"  {a:<28}  {stats['avg_rounds_won_a']:.2f}")
    print(f"  {b:<28}  {stats['avg_rounds_won_b']:.2f}")
    print(f"  {'Draw rounds (avg)':<28}  {stats['avg_draw_rounds']:.2f}")

    print("\n  CLEAN PUNCHES LANDED  (avg per round)")
    print(f"  {a:<28}  {stats['avg_punches_per_round_a']:.2f}")
    print(f"  {b:<28}  {stats['avg_punches_per_round_b']:.2f}")

    print("\n  ROUND DOMINANCE  (rounds 1–4 vs 5–10)")
    print(f"  {'Phase':<14}  {a[:18]:>18}   {b[:18]:>18}")
    print(f"  {'─' * 54}")
    print(f"  {'Early (1–4)':<14}  {stats['early_win_pct_a']:>17.1f}%   {stats['early_win_pct_b']:>17.1f}%")
    print(f"  {'Late  (5–10)':<14}  {stats['late_win_pct_a']:>17.1f}%   {stats['late_win_pct_b']:>17.1f}%")

    print("\n  DISTANCE DISTRIBUTION  (% of all exchanges)")
    bar_w = 30
    for label, dist_key in [("Inside ", INSIDE), ("Mid    ", MID), ("Outside", OUTSIDE)]:
        pct = dp[dist_key]
        filled = int(round(pct / 100 * bar_w))
        bar = "█" * filled + "░" * (bar_w - filled)
        print(f"  {label}  {bar}  {pct:5.1f}%")

    print(f"\n{sep}\n")
