from dataclasses import dataclass
from .enums import Colors

@dataclass(eq=False, frozen=True)
class Marble:
    color: Colors

    def __str__(self) -> str:
        return f"{self.color.name.capitalize()} marble {id(self)}"
