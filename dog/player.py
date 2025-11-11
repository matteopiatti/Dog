from dataclasses import dataclass, field
from typing import List, Iterable, Optional, Set
from .cards import Card
from .marble import Marble

@dataclass
class Player:
    name: str
    marbles: List["Marble"]
    hand: List["Card"] = field(default_factory=list)
    marbles_in_play: Set["Marble"] = field(default_factory=set)

    def __str__(self) -> str:
        cards_str = ", ".join(str(c) for c in self.hand)
        marbles_str = ", ".join(str(m) for m in self.marbles)
        return f"Player({self.name}, hand=[{cards_str}], marbles=[{marbles_str}])"

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

    def get_marble(self, marble: "Marble") -> Optional[int]:
        for i, m in enumerate(self.marbles):
            if m is marble:
                return i + 1
        return None

    def get_free_marble(self) -> Optional["Marble"]:
        for marble in self.marbles:
            if marble not in self.marbles_in_play:
                self.marbles_in_play.add(marble)
                return marble
        return None
