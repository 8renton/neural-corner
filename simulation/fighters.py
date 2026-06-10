from dataclasses import dataclass
from simulation.styles import StyleProfile, build_philly_shell, build_soviet_style


@dataclass
class Fighter:
    name: str
    height_cm: int
    style: StyleProfile

    def reach_advantage(self, opponent: "Fighter") -> float:
        """Positive value means this fighter is taller with longer reach."""
        return (self.height_cm - opponent.height_cm) / 10.0

    def __repr__(self) -> str:
        return f"{self.name} ({self.height_cm}cm, {self.style.name})"


def create_philly_shell_fighter() -> Fighter:
    return Fighter(
        name="Fighter A",
        height_cm=170,
        style=build_philly_shell(),
    )


def create_soviet_style_fighter() -> Fighter:
    return Fighter(
        name="Fighter B",
        height_cm=183,
        style=build_soviet_style(),
    )
