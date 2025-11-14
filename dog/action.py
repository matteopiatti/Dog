from dataclasses import dataclass
from dog.cards import Card
from dog.player import Player
from dog.marble import Marble
from .enums import MoveKind

@dataclass
class Action:
    player: Player
    card: Card
    kind: MoveKind
    marble: Marble
    swap_marble: Marble | None
    swap_player: Player | None
    steps: int | None