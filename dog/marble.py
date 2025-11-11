from dataclasses import dataclass
from .enums import Colors

@dataclass(frozen=True)
class Marble:
    color: Colors

    def __str__(self) -> str:
        return f"{self.color.name.capitalize()} marble"
