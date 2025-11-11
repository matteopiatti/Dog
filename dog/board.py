from dog.player import Player
from dog.marble import Marble

class Board:
    NUM_FIELDS = 64
    START_FIELDS = {
        0: 0,
        1: 16,
        2: 32,
        3: 48,
    }

    def __init__(self, num_players:int):
      self.track = [None] * self.NUM_FIELDS
      self.home = {pid: [None] * 4 for pid in range(num_players)}
      self.start_fields: dict[int, int] = {
        self.START_FIELDS[pid]: pid for pid in range(num_players)
      }
      self.occupied_fields = set()

    def get_player_marbles(self, player: "Player"):
      marbles = []
      for marble in self.track:
        if marble is not None and marble.color == player.color:
          marbles.append(marble)
      return marbles
        
    def start_marble(self, player_idx, player: "Player"):
      player_startfield = self.START_FIELDS[player_idx]
      marble = player.get_free_marble()
      if marble and self.track[player_startfield] is None:
          self.track[player_startfield] = marble
          player.marbles_in_play.add(marble)
          self.occupied_fields.add(player_startfield)
          return True
      
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
    
    def is_valid_move(self, marble, step):
      if marble not in self.track:
          return False
      current_pos = next(i for i, m in enumerate(self.track) if m is marble)
      for s in range(1, abs(step)+1):
          intermediate_pos = (current_pos + (s if step > 0 else -s)) % self.NUM_FIELDS
          if intermediate_pos in self.occupied_fields:
              return False
      return True