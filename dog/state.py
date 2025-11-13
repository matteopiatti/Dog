from collections.abc import Sequence
from dataclasses import dataclass, field
from .cards import Deck, Card
from .player import Player
from .board import Board

@dataclass
class GameState:
  players: Sequence[Player]
  board: "Board"
  deck: "Deck"
  discard_pile: list["Card"] = field(default_factory=list)
  draw_size: int = 0
  current_player: int = 0
  finished: bool = False
  winner: int | None = None