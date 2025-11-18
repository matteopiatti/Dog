from collections.abc import Sequence
from dataclasses import dataclass, field
from agent.agent import Agent
from .cards import Deck, Card
from .objects import Player, Action
from .board import Board
from .enums import GamePhase
from copy import deepcopy


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
    teams: list[tuple[Player, Player]] = field(default_factory=list)
    switch_cards = []
    num_rounds: int = 0
    winner: tuple[Player, Player] | None = None
    agents: dict[Player, Agent] = field(default_factory=dict)

    @property
    def next_player(self):
        return self.players[
            (self.players.index(self.current_player) + 1) % len(self.players)
        ]

    @property
    def empty_hands(self):
        return all(len(player.hand) == 0 for player in self.players)

    def advance_player(self):
        self.current_player = self.next_player

    def reset_actions(self):
        self.cp_actions.clear()

    def advance_round(self):
        self.new_round()
        if self.draw_size <= 2:
            self.draw_size = 6
        else:
            self.draw_size -= 1

    def new_round(self):
        self.num_rounds += 1
        i = self.players.index(self.last_started_player)
        self.current_player = self.players[(i + 1) % len(self.players)]
        self.reset_actions()
        self.last_started_player = self.current_player

    def teammate(self, player: Player) -> Player:
        for team in self.teams:
            if player in team:
                return team[0] if team[1] is player else team[1]
        return None

    def player_finished(self, player: Player) -> bool:
        if self.board.player_finished_marbles(player) == 4:
            return True
        return False

    def check_winner(self):
        for team in self.teams:
            if self.player_finished(team[0]) and self.player_finished(team[1]):
                self.finished = True
                self.winner = team
                return team
        return None

    def clone(self):
        return deepcopy(self)
