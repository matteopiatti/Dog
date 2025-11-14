from .cli_helpers import get_printable_card, middle_row, cell_str, print_action_line, get_printable_marble, print_action_selection_line
from .enums import GamePhase
from .action import Action

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
    print(f"{l}{middle_row(i, top, state)}{r}")
  print(" ".join(bottom))

def print_no_actions() -> None:
  print("\033[s")
  print("\033[1;50H\033[4mNo possible actions available. Folding hand.\033[0m")
  print("\033[u")
  input("Press Enter to continue...")

def print_actions(actions) -> None:
  print("\033[s")
  print("\033[1;50H\033[4mPossible actions:\033[0m")
  for idx, action in enumerate(actions):
    print_action_line(action, idx)
  print("\033[u")

def select_split_action(action: Action, allowed_steps: dict) -> None:
  print("\033[s")
  print("\033[1;50H\033[4mSplit Action Selected. Allowed steps for each marble:\033[0m")
  options = []
  i = 0
  for marble, steps in allowed_steps.items():
    marble_str = get_printable_marble(marble, action.player)
    for step in steps:
      i += 1
      options.append((marble, step))
      print(f"\033[{i+1};50H{i}:  {marble_str}: {step}")
  print("\033[u")
  while True:
    choice = input("Select marble and steps: ")
    if choice.isdigit():
      choice_idx = int(choice) - 1
      if 0 <= choice_idx < len(options):
        selected_marble, selected_steps = options[choice_idx]
        return selected_marble, selected_steps
    error_print("Invalid choice. Please try again.")

def select_action(state) -> Action:
  while True:
    choice = input("Select an Action: ")
    if choice.isdigit():
      choice_idx = int(choice) - 1
      if 0 <= choice_idx < len(state.cp_actions):
        return state.cp_actions[choice_idx]
    error_print("Invalid choice. Please try again.")