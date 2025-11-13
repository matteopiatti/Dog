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

    def get_player_marbles(self, player: "Player"):
      marbles = []
      for marble, p in self.track:
        if marble is not None and marble.color == player.color:
          marbles.append(marble)
      return marbles
    
    def get_free_player_marble(self, player: "Player") -> Marble | None:
      for marble in player.marbles:
        if marble not in self.track:
          return marble
      return None

    def player_marbles_in_play(self, state, player: "Player"):
      marbles = []
      for marble in player.marbles:
        if any(m == marble for m, _ in state.board.track):
          marbles.append(marble)
      return marbles
        
    def player_has_startable_marble(self, player: "Player") -> bool:
      for marble in player.marbles:
        if marble not in self.track and marble not in self.home[player]:
          return True
      return False

    # should be in move.py
    def start_marble(self, player_idx, player: "Player"):
      player_startfield = self.START_FIELDS[player_idx]
      marble = player.get_free_marble()
      if marble and self.track[player_startfield] is None:
          self.track[player_startfield] = marble
          player.marbles_in_play.add(marble)
          self.occupied_fields.add(player_startfield)
          return True
      
    # should be in move.py
    def move_marble(self, player: "Player", marble: "Marble", steps: int, players: list["Player"]):
      if marble not in player.marbles_in_play:
          return False

      current_pos = next(i for i, m in enumerate(self.track) if m is marble)
      new_pos = (current_pos + steps) % self.NUM_FIELDS

      if current_pos in self.start_fields:
          self.occupied_fields.discard(current_pos)

      if self.track[new_pos] is None:
          self.track[current_pos] = None
          self.track[new_pos] = marble
          return True
      else:
        occupying_marble = self.track[new_pos]
        self.track[current_pos] = None
        self.track[new_pos] = marble
        occupying_player = None
        for p in players:
          if occupying_marble in p.marbles:
            occupying_player = p
            break
        if occupying_player:
          occupying_player.marbles_in_play.discard(occupying_marble)
        return True
    
    # IMPLEMENTED; should be in rules.py
    def is_valid_move(self, marble, step):
      if marble not in self.track:
          return False
      current_pos = next(i for i, m in enumerate(self.track) if m is marble)
      for s in range(1, abs(step)+1):
          intermediate_pos = (current_pos + (s if step > 0 else -s)) % self.NUM_FIELDS
          if intermediate_pos in self.occupied_fields:
              return False
      return True