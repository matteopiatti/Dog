from typing import Sequence
from dog.player import Player
from .cli_helpers import get_printable_card, middle_row, cell_str, print_action_line
from .enums import GamePhase
from .move import Action

# this should render whatever is the step. It should not contain game logic.
def render(state) -> None:
  clear_screen()
  print_board(state)
  print_hand(state.current_player)
  if state.phase == GamePhase.PLAY:
    print_actions(state.cp_actions)


def clear_screen() -> None:
  print("\033c", end="")

def error_print(msg: str) -> None:
  print(f"\033[1;41m{msg}\033[0m")

def print_hand(player) -> str:
  cards = [get_printable_card(card) for card in player.hand]
  hand_str = " ".join(cards)
  print(f"{player.color.value}{player.name}'s hand: {hand_str}\033[0m")

def print_board(state) -> None:
  top = [cell_str(state, i) for i in range(0, 17)][::-1]
  left = [cell_str(state, i) for i in range(17, 32)]
  bottom = [cell_str(state, i) for i in range(32, 49)]
  right = [cell_str(state, i) for i in range(49, 64)][::-1]
                                                  
  print(" ".join(top))
  for i in range(15):
    l = left[i]
    r = right[i]
    print(f"{l}{middle_row(i, top)}{r}")
  print(" ".join(bottom))

def print_no_actions() -> None:
  print("\033[s")
  print("\033[1;50H\033[4mNo possible actions available.\033[0m")
  print("\033[u")
  input("Press Enter to continue...")

def print_actions(actions) -> None:
  print("\033[s")
  print("\033[1;50H\033[4mPossible actions:\033[0m")
  for idx, action in enumerate(actions):
    marble_str = f"Marble {action.player.get_marble(action.marble)}"
    if action.marble is None:
        marble_str = "New Marble"
    steps_str = f"Steps {action.steps}" if action.steps is not None else ""
    print_action_line(idx, get_printable_card(action.card), marble_str, steps_str)
  print("\033[u")

def select_action(state) -> Action:
  while True:
    print("Select an Action:")
    for idx, option in enumerate(state.cp_actions):
      print(f"  {idx + 1}: {get_printable_card(option.card)}")
    choice = input("Select an option: ")
    if choice.isdigit():
      choice_idx = int(choice) - 1
      if 0 <= choice_idx < len(state.cp_actions):
        return state.cp_actions[choice_idx]
    error_print("Invalid choice. Please try again.")