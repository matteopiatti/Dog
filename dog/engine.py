from .state import GameState
from .board import Board
from .cards import Deck
from .player import Player
from .marble import Marble
from .enums import Colors
from .cli import clear_screen, print_board, print_hand, prompter, moves_prompter
from .move import generate_moves, move_action

class Engine:
    def __init__(self):
        pass
    
    @staticmethod
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
      board = Board(num_players=num_players)
      return GameState(players=players, board=board, deck=deck)
    
    @staticmethod
    def start_game(state: GameState) -> GameState:  
      print("Welcome to the Dog game!")
      input("Press Enter to start the game...")
      while not state.finished:
        Engine.start_round(state)
      return state
    
    @staticmethod
    def start_round(state: GameState) -> GameState:
      if state.draw_size <= 1:
        state.draw_size = 6
      else:
        state.draw_size -= 1
      for player in state.players:
        player.receive_hand(state.deck.deal(state.draw_size))
      clear_screen()
      print_board(state.board, state.players)
      Engine.start_turn(state)
      input("Press Enter to continue...")

    def start_turn(state: GameState) -> GameState:
      cp = state.players[state.current_player]

      if Engine.has_round_ended(state):
        print("Round has ended. Starting new round...")
        Engine.new_round(state)
        return state

      if cp.hand == []:
        print(f"{cp.color.value}{cp.name} has no cards left. Skipping turn.\033[0m")
        Engine.next_turn(state)
        return state

      print(f"\n{cp.color.value}{cp.name}'s turn:\033[0m")
      print_hand(cp)
      actions = generate_moves(state, cp, cp.hand[0])

      if not actions:
        print("No legal moves available. Hand folded.")
        input("Press Enter to end turn...")
        cp.fold_hand()
        Engine.next_turn(state)
        return state

      player_action = moves_prompter("Select a card to play:", actions, cp)
      print(f"You selected: {cp.hand[player_action.card_idx]}")
      move_action(state, player_action)
      Engine.next_turn(state)
      return state

    def next_turn(state: GameState) -> GameState:
      state.current_player = (state.current_player + 1) % len(state.players)
      clear_screen()
      print_board(state.board, state.players)
      Engine.start_turn(state)
      return state
    
    @staticmethod
    def new_round(state: GameState) -> GameState:
      return state
    
    def has_round_ended(state: GameState) -> bool:
      return all(player.hand == [] for player in state.players)