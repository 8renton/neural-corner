from dataclasses import dataclass

INSIDE = "inside"
MID = "mid"
OUTSIDE = "outside"

DISTANCE_STATES = [INSIDE, MID, OUTSIDE]
DISTANCE_INDEX = {INSIDE: 0, MID: 1, OUTSIDE: 2}


@dataclass
class StyleProfile:
    name: str

    # Probability of landing a clean punch at each distance state
    offense: dict  # {distance: float}

    # Probability of successfully defending at each distance state
    defense: dict  # {distance: float}

    # Bonus punch probability when countering a committed attack
    counter_rate: float

    # Jab effectiveness for controlling range
    jab_control: float

    # Ability to press through opposition and close distance
    pressure_ability: float

    # Preferred fighting distance — where the style performs best
    preferred_distance: str

    # Success weight when trying to close to inside range
    clinch_ability: float

    # Success weight when trying to maintain or create outside range
    distance_creation: float


def build_philly_shell() -> StyleProfile:
    return StyleProfile(
        name="Philly Shell",
        offense={INSIDE: 0.62, MID: 0.48, OUTSIDE: 0.28},
        defense={INSIDE: 0.74, MID: 0.62, OUTSIDE: 0.48},
        counter_rate=0.55,
        jab_control=0.32,
        pressure_ability=0.65,
        preferred_distance=INSIDE,
        clinch_ability=0.62,
        distance_creation=0.35,
    )


def build_soviet_style() -> StyleProfile:
    return StyleProfile(
        name="Soviet Style",
        offense={INSIDE: 0.38, MID: 0.55, OUTSIDE: 0.67},
        defense={INSIDE: 0.44, MID: 0.60, OUTSIDE: 0.73},
        counter_rate=0.32,
        jab_control=0.72,
        pressure_ability=0.42,
        preferred_distance=OUTSIDE,
        clinch_ability=0.33,
        distance_creation=0.68,
    )
