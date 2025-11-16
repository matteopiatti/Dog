from dataclasses import dataclass, field
from typing import List, Iterable, Optional, Set
from .cards import Card
from .enums import Colors, MoveKind


@dataclass(eq=False, frozen=True)
class Marble:
    color: Colors


@dataclass(eq=False)
class Player:
    name: str
    marbles: List[Marble]
    hand: List["Card"] = field(default_factory=list)

    @property
    def color(self):
        return self.marbles[0].color if self.marbles else None

    def receive_hand(self, cards: Iterable["Card"]) -> None:
        self.hand.extend(cards)

    def receive_card(self, card: "Card") -> None:
        self.hand.append(card)

    def play_card(self, card: "Card") -> Optional["Card"]:
        try:
            self.hand.remove(card)
        except ValueError:
            return None
        return card

    def fold(self) -> None:
        self.hand.clear()


@dataclass
class Action:
    player: Player
    card: Card
    kind: MoveKind
    marble: Marble | None = None
    swap_marble: Marble | None = None
    swap_player: Player | None = None
    steps: int | None = None
