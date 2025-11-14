from collections.abc import Sequence
from dataclasses import dataclass, field
from .cards import Deck, Card
from .objects import Player, Action
from .board import Board
from .enums import GamePhase

@dataclass
class GameState:
  players: Sequence[Player]
  board: "Board"
  deck: "Deck"
  discard_pile: list["Card"] = field(default_factory=list)
  draw_size: int = 0
  current_player: Player = None
  last_started_player: Player = None
  finished: bool = False
  winner: Player | None = None
  phase: GamePhase = GamePhase.DEAL
  cp_actions: list["Action"] = field(default_factory=list)

  @property
  def next_player(self):
    return self.players[(self.players.index(self.current_player) + 1) % len(self.players)]
  
  @property
  def empty_hands(self):
    return all(len(player.hand) == 0 for player in self.players)