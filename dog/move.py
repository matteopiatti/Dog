from dataclasses import dataclass
from dog.cards import Card
from dog.player import Player
from dog.marble import Marble
from .enums import MoveKind

# action doesn't need move kind if we have movekind on each Card
@dataclass
class Action:
    # player_idx: int
    # card_idx: int
    player: Player
    card: Card
    kind: MoveKind
    marble: Marble
    steps: int | None

def start_action(state, action):
  if action.kind == MoveKind.START:
    start_marble(state)
  elif action.kind == MoveKind.MOVE:
    # move marble
    print("Moving marble...")
  
  ## implement 7, split move and jack, swap move
  
def start_marble(state):
  startfield = state.board.start_fields[state.current_player]
  marble = state.board.get_free_player_marble(state.current_player)
  if marble and startfield not in state.board.occupied_fields:
      state.board.track[startfield] = (marble, state.current_player)
      state.board.occupied_fields.add(startfield)
      return True
  
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