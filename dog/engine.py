from .state import GameState
from .board import Board
from .cards import Deck
from .player import Player
from .marble import Marble
from .enums import Colors
from .cli import clear_screen, print_board, print_hand, print_no_actions, select_action
from .enums import GamePhase
from dog.rules import legal_actions
from .move import start_action

def setup_game(num_players: int) -> GameState:
      deck = Deck()
      deck.shuffle()
      players = [
          Player(
              name=f"Player {color.name.capitalize()}",
              marbles=[Marble(color) for _ in range(4)]
          )
          for color in list(Colors)[:num_players]
      ]
      board = Board(players=players)
      return GameState(players=players, board=board, deck=deck, current_player=players[0], last_started_player=players[0])

def step(state: GameState) -> GameState:
  if state.phase == GamePhase.DEAL:
    deal_cards(state)
    state.phase = GamePhase.TURN
  elif state.phase == GamePhase.TURN:
    legal_actions(state)
    if not state.cp_actions:
      print_no_actions()
      state.current_player = state.next_player
    else:
      state.phase = GamePhase.PLAY
  elif state.phase == GamePhase.PLAY:
    action = select_action(state)
    start_action(state, action)
    state.current_player.play_card(action.card)
    state.cp_actions.clear()
    state.current_player = state.next_player
    state.phase = GamePhase.TURN
    input("Press Enter to continue...")

def deal_cards(state: GameState) -> GameState:
  if state.draw_size <= 1:
    state.draw_size = 6
    state.current_player = state.next_player
    state.last_started_player = state.next_player
  else:
    state.draw_size -= 1
  for player in state.players:
    player.receive_hand(state.deck.deal(state.draw_size))


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