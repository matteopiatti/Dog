from dataclasses import dataclass, field
from typing import List, Iterable, Optional, Set
from .cards import Card
from .marble import Marble

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

    def fold_hand(self) -> None:
        self.hand.clear()

    # IMPLEMENTED as in not needed
    def get_marble(self, marble: "Marble") -> Optional[int]:
        for i, m in enumerate(self.marbles):
            if m is marble:
                return i + 1
        return None

    # IMPLEMENTED; should be in board.py
    def get_free_marble(self) -> Optional["Marble"]:
        for marble in self.marbles:
            if marble not in self.marbles_in_play:
                self.marbles_in_play.add(marble)
                return marble
        return None
