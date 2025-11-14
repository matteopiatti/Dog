from dog.player import Player
from dog.marble import Marble

class Board:
    NUM_FIELDS = 64
    START_FIELD_NUMBERS = {
        0: 0,
        1: 16,
        2: 32,
        3: 48,
    }

    def __init__(self, players: list["Player"]):
      self.track = [(None, None)] * self.NUM_FIELDS
      self.home = {player: [None] * 4 for player in players}
      self.start_fields: dict[Player, int] = {
        player: self.START_FIELD_NUMBERS[idx] for idx, player in enumerate(players)
      }
      self.occupied_fields = set()
      # self.home[players[0]][3] = Marble(players[0].color)  # For testing purposes
      # self.home[players[1]][2] = Marble(players[1].color)  # For testing purposes
      # self.home[players[2]][1] = Marble(players[2].color)  # For testing purposes
      # self.home[players[3]][0] = Marble(players[3].color)  # For testing purposes
    
    def marble_can_move_home(self, marble: Marble, player: "Player", steps) -> bool:
      startfield = self.start_fields[player]
      if startfield in self.occupied_fields:
        return False
      pos = next(i for i, (m, p) in enumerate(self.track) if m is marble)
      distance_to_start = (startfield - pos) % self.NUM_FIELDS
      distance_to_first_marble = next((i for i, v in enumerate(self.home[player]) if v is not None), 4)
      return steps <= distance_to_start + distance_to_first_marble and steps > distance_to_start

    def get_free_player_marble(self, player: "Player") -> Marble | None:
      for marble in player.marbles:
        if marble not in self.player_marbles_in_play(player):
          return marble
      return None

    def player_marbles_in_play(self, player: "Player"):
      marbles = []
      for marble in player.marbles:
        if any(m == marble for m, _ in self.track):
          marbles.append(marble)
      return marbles
        
    def player_has_startable_marble(self, player: "Player") -> bool:
      for marble in player.marbles:
        if marble not in self.track and marble not in self.home[player]:
          return True
      return False