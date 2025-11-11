import Player

class Board:
    NUM_FIELDS = 64
    START_FIELDS = {
        0: 0,
        1: 16,
        2: 32,
        3: 48,
    }

    def __init__(self, players):
        self.track = [None] * self.NUM_FIELDS
        self.players = players
        self.home = {
            player: [None] * 4 for player in players
        }
        self.start_fields = {
            self.START_FIELDS[idx]: player
            for idx, player in enumerate(players)
        }

    def is_start_field(self, idx: int):
        player = self.start_fields.get(idx)
        if player is not None:
            return player.get_color()
        return None
    
    def is_field_free(self, idx: int) -> bool:
        return self.track[idx] is None

    def start_marble(self, player: Player):
        start_idx = self.START_FIELDS[self.players.index(player)]
        if self.is_field_free(start_idx):
            self.track[start_idx] = (player, player.get_marble())
        else:
            print(f"Start field for {player.name} is occupied!")


    def move_marble(self, player: Player, steps: int, marble=None):
        pos = self.get_marble_position(player, marble)
        target_idx = None
        end = self.NUM_FIELDS - pos
        if steps < end:
            target_idx = pos + steps
        else:
            target_idx = steps - end
        self.track[target_idx] = self.track[pos]
        self.track[pos] = None

    def get_player_marbles(self, player: Player):
        marbles = []
        for idx, cell in enumerate(self.track):
            if cell is not None and cell[0] == player:
                marbles.append((idx, cell[1]))
        for idx, cell in enumerate(self.home[player]):
            if cell is not None:
                marbles.append((self.NUM_FIELDS + idx, cell[1]))
        return marbles
    
    def get_marble_position(self, player: Player, marble):
        for idx, cell in enumerate(self.track):
            if cell is not None and cell[0] == player and cell[1] == marble:
                return idx
        for idx, cell in enumerate(self.home[player]):
            if cell is not None and cell == marble:
                return self.NUM_FIELDS + idx
        return None
    
    def player_can_move(self, player: Player, steps: int):
        marbles = self.get_player_marbles(player)
        for idx, marble in marbles:
            end = self.NUM_FIELDS - idx
            target_idx = None
            if steps < end:
                target_idx = idx + steps
            else:
                target_idx = steps - end

            if self.is_field_free(target_idx):
                    return True
        return False
    
    def get_movable_marbles(self, player: Player, steps: int):
        marbles = self.get_player_marbles(player)
        movable = []
        for idx, marble in marbles:
            end = self.NUM_FIELDS - idx
            target_idx = None
            if steps < end:
                target_idx = idx + steps
            else:
                target_idx = steps - end

            if self.is_field_free(target_idx):
                    movable.append(marble)
        return movable

    def is_field_free(self, idx: int) -> bool:
        return True

    def render(board):
      t = board.track

      def cell_str(i):
          if t[i] is None:
              color = board.is_start_field(i)
              if color:
                  return color+"◯\033[0m"
              return "."
          p, pid = t[i]
          marble_id = None
          for marble in p.marbles_in_play:
              if board.get_marble_position(p, marble) == i:
                  marble_id = p.marbles.index(marble) + 1
                  break
          symbols = ["❶", "❷", "❸", "❹"]
          return p.get_color() + symbols[marble_id - 1] + "\033[0m"

      top = [cell_str(i) for i in range(0, 17)][::-1]
      right = [cell_str(i) for i in range(17, 32)][::-1]
      bottom = [cell_str(i) for i in range(32, 49)][::-1]
      left = [cell_str(i) for i in range(49, 64)][::-1]

      print(" ".join(top))
      for i in range(15):
          l = left[i] if i < len(left) else " "
          r = right[i] if i < len(right) else " "
          c = ' ' * (len(top)*2 - 3)
          if i == 0 or i == 14:
              c = " ❂" + " " * (len(top)*2 - 7) + "❂ "
          elif i == 1 or i == 13:
              c = "   ❂" + " " * (len(top)*2 - 11) + "❂   "
          elif i == 2 or i == 12:
              c = "     ❂" + " " * (len(top)*2 - 15) + "❂     "
          elif i == 3 or i == 11:
              c = "       ❂" + " " * (len(top)*2 - 19) + "❂       "
          elif i == 7:
              c = "              DOG              "
          print(f"{l}{c}{r}")
      print(" ".join(bottom[::-1]))