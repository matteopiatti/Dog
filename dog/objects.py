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
    marbles: List[int]
    hand: List["Card"] = field(default_factory=list)
    # could technically be derived from marbles + board state
    marbles_in_play: Set["Marble"] = field(default_factory=set)

    @property
    def color(self):
        return self.marbles[0].color if self.marbles else None

    def receive_hand(self, cards: Iterable["Card"]) -> None:
        self.hand.extend(cards)

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
    marble: Marble
    swap_marble: Marble | None
    swap_player: Player | None
    steps: int | None