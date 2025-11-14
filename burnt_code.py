
# engine should not handle any I/O. Move all print statements to cli or main.
# engine should not be a class
class Engine:
    def __init__(self):
        pass

    @staticmethod
    # should be in main
    def start_game(state: GameState) -> GameState:  
      while not state.finished:
        Engine.start_round(state)
      return state
    
    @staticmethod
    # should be inside step
    def start_round(state: GameState) -> GameState:
      if state.draw_size <= 1:
        state.draw_size = 6
      else:
        state.draw_size -= 1
      for player in state.players:
        player.receive_hand(state.deck.deal(state.draw_size))
      
      state.phase = state.PHASE.TURN

    # should be inside step
    def start_turn(state: GameState) -> GameState:
      cp = state.players[state.current_player]

      if Engine.has_round_ended(state):
        print("Round has ended. Starting new round...")
        Engine.new_round(state)
        return state

      if cp.hand == []:
        print(f"{cp.color.value}{cp.name} has no cards left. Skipping turn.\033[0m")
        return state

      print(f"\n{cp.color.value}{cp.name}'s turn:\033[0m")
      print_hand(cp)
      actions = generate_moves(state, cp, cp.hand[0])

      if not actions:
        print("No legal moves available. Hand folded.")
        input("Press Enter to end turn...")
        cp.fold_hand()
        return state

      player_action = moves_prompter("Select a card to play:", actions, cp)
      print(f"You selected: {cp.hand[player_action.card_idx]}")
      move_action(state, player_action)
      return state

    # should be inside step
    def next_turn(state: GameState) -> GameState:
      state.current_player = (state.current_player + 1) % len(state.players)
      clear_screen()
      print_board(state.board, state.players)
      Engine.start_turn(state)
      return state
    
    @staticmethod
    # nope
    def new_round(state: GameState) -> GameState:
      return state
    
    # state should know that
    def has_round_ended(state: GameState) -> bool:
      return all(player.hand == [] for player in state.players)
    
    # also state should know that
    def turn_has_ended(state: GameState) -> bool:
      cp = state.players[state.current_player]
      return cp.hand == []
   

 # IMPLEMENTED should be in move.py
    def start_marble(self, player_idx, player: "Player"):
      player_startfield = self.START_FIELDS[player_idx]
      marble = player.get_free_marble()
      if marble and self.track[player_startfield] is None:
          self.track[player_startfield] = marble
          player.marbles_in_play.add(marble)
          self.occupied_fields.add(player_startfield)
          return True
      
    # IMPLEMENTED; should be in move.py
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

# some part of this generate_moves ought to be in rules.py
# def generate_moves(state: "GameState", current_player: "Player", card: "Card") -> list["Action"]:
#   actions = []
#   player_marbles = state.board.get_player_marbles(current_player)

#   for card in current_player.hand:
#     if card.rank in [CardType.ACE, CardType.KING, CardType.JOKER]:
#       if len(current_player.marbles_in_play) < 4 and state.board.home[state.current_player].count(None) > 0:
#         actions.append(
#             Action(
#                 player_idx=state.players.index(current_player),
#                 card_idx=current_player.hand.index(card),
#                 card=card,
#                 kind=MoveKind.START,
#                 marble=None,
#                 steps=None,
#             )
#         )
#     for marble in player_marbles:
#       steps = []
#       if card.rank == CardType.ACE:
#         steps = [1, 11]
#       elif card.rank == CardType.KING:
#         steps = [13]
#       elif card.rank == CardType.QUEEN:
#         steps = [12]
#       elif card.rank == CardType.TEN:
#         steps = [10]
#       elif card.rank == CardType.NINE:
#         steps = [9]
#       elif card.rank == CardType.EIGHT:
#         steps = [8]
#       elif card.rank == CardType.SEVEN:
#         steps = [7]
#       elif card.rank == CardType.SIX:
#         steps = [6]
#       elif card.rank == CardType.FIVE:
#         steps = [5]
#       elif card.rank == CardType.FOUR:
#         steps = [4, -4]
#       elif card.rank == CardType.THREE:
#         steps = [3]
#       elif card.rank == CardType.TWO:
#         steps = [2]
#       elif card.rank == CardType.JOKER:
#         steps = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
      
#       for step in steps:
#         if state.board.is_valid_move(marble, step):
#           actions.append(
#               Action(
#                   player_idx=state.players.index(current_player),
#                   card_idx=current_player.hand.index(card),
#                   card=card,
#                   kind=MoveKind.MOVE,
#                   marble=marble,
#                   steps=step,
#               )
#           )
#   return actions

# # here move action calls the board but should actually do the whole move
# def move_action(state: "GameState", action: "Action") -> "GameState":
#   player = state.players[action.player_idx]
#   card = player.hand[action.card_idx]

#   if action.kind == MoveKind.START:
#     state.board.start_marble(action.player_idx, player)
#     player.play_card(card)
#   elif action.kind == MoveKind.MOVE:
#     state.board.move_marble(player, action.marble, action.steps, state.players)
#     player.play_card(card)

#   return state



    # mip = state.board.player_marbles_in_play(state, state.current_player)
    # for card in state.current_player.hand:
    #   if card.
    #     for marble in state.current_player.marbles:
    #         if marble not in state.board.track:
    #            start_field = state.board.start_fields[state.players.index(state.current_player)]
    #           if start_field not in state.board.occupied_fields:
    #               actions.append(Action(state.current_player)
    #               break

    #         elif marble in state.board.track:
    #             ## marble is in play
    #     for marble in mip:
    #         if marble not in state.board.track:
    #             start_field = state.board.start_fields[state.players.index(state.current_player)]
    #             if start_field not in state.board.occupied_fields:
    #                 actions.append((card, None))
    #                 break
    #     for marble in mip:
    #         if marble not in state.board.track:
    #             continue
    #         current_pos = next(i for i, m in enumerate(state.board.track) if m is marble)
    #         for s in range(1, abs(card.value)+1):
    #             intermediate_pos = (current_pos + (s if card.value > 0 else -s)) % state.board.NUM_FIELDS
    #             if intermediate_pos in state.board.occupied_fields:
    #                 break
    #         else:
    #             actions.append((card, marble))