# Neural Corner

A Python boxing simulation engine that models stylistic matchups between AI fighters using probabilistic exchanges, round scoring, and reinforcement learning.

Neural Corner is designed to feel like a **computational sports science tool**, not a video game. No health bars. No damage systems. Just distance control, punch selection probability, defensive success rates, and round-based scoring — the actual mechanics of boxing.

---

## Fighters

| | Fighter A | Fighter B |
|---|---|---|
| **Style** | Philly Shell | Soviet Style |
| **Height** | 170 cm | 183 cm |
| **Strengths** | Close-range defense, counter punching, slipping | Jab control, distance management, volume scoring |
| **Weaknesses** | Long range control, jab vs taller opponents | Inside exchanges, pressure defense |

The 13 cm height difference affects reach advantage, jab success probability, and distance control effectiveness at each range.

---

## How It Works

Each fight runs over 10 rounds of 8–12 exchanges. No health points — every exchange calculates:

- Who controls distance (inside / mid / outside)
- Who lands clean punches (based on style offense/defense rates + reach + positional advantage)
- Counter punch windows when an opponent commits
- Distance shifts from movement choices

Rounds are scored by punch differential. Fights by majority rounds.

---

## Project Structure

```
neural-corner/
├── main.py                        # Batch simulation (500 fights, stats report)
├── simulation/
│   ├── styles.py                  # StyleProfile — offense/defense rates per distance
│   ├── fighters.py                # Fighter dataclass + factories
│   └── engine.py                  # Exchange, round, and fight simulation
├── ai/
│   ├── decision_policy.py         # Rule-based action selection (weighted probabilities)
│   ├── agent.py                   # Stateful rule-based fighter agent
│   └── trained_agent.py           # PPO-backed agent (post-training)
├── training/
│   ├── boxing_env.py              # Gymnasium environment
│   ├── train.py                   # Curriculum training script
│   └── trained_models/            # Saved .zip model files (after training)
├── visualization/
│   ├── fight_viewer.py            # Pygame real-time fight renderer
│   └── run_fight.py               # Entry point for the fight viewer
└── utils/
    └── stats.py                   # Aggregate stats + analytics report
```

---

## Installation

```bash
git clone https://github.com/8renton/neural-corner.git
cd neural-corner
pip install -r requirements.txt
```

---

## Usage

### Run a batch simulation (500 fights, analytics report)

```bash
python main.py
```

### Watch a live fight in Pygame

```bash
python visualization/run_fight.py
```

Controls: `SPACE` to skip scenes · `ESC` to quit

### Train the AI fighters

```bash
python -m training.train
```

Runs a 3-phase curriculum (~15–30 min on a modern laptop, no GPU needed):

| Phase | What happens |
|---|---|
| **1 — Style Drills** | Each fighter trains against a scripted sparring partner |
| **2 — Film Study** | Each fighter trains against the opponent's Phase 1 policy |
| **3 — Self-Play** | Both fighters adapt to each other over 3 alternating iterations |

### Watch a trained fight

```bash
python visualization/run_fight.py --trained
```

---

## Sample Output

```
══════════════════════════════════════════════════════════════
  NEURAL CORNER  ·  BOXING SIMULATION ANALYTICS REPORT
══════════════════════════════════════════════════════════════
  Fights simulated : 500

  FIGHT OUTCOMES
  Fighter A (Philly Shell)     63.8%    319 fights
  Fighter B (Soviet Style)     19.8%     99 fights
  Draws                        16.4%     82 fights

  DISTANCE DISTRIBUTION
  Inside   ████████████░░░░░░░░░░░░░░░░░░   41.6%
  Mid      ██████████████░░░░░░░░░░░░░░░░   47.7%
  Outside  ███░░░░░░░░░░░░░░░░░░░░░░░░░░░   10.7%
══════════════════════════════════════════════════════════════
```

---

## Roadmap

- [ ] Additional fighter archetypes (pressure southpaw, volume boxer, out-fighter)
- [ ] Style drift over rounds (fatigue modeled as decision degradation)
- [ ] Ring control heatmaps
- [ ] Multi-fighter tournament mode
