from typing import Sequence

from dog.player import Player


def clear_screen() -> None:
  print("\033c", end="")

def error_print(msg: str) -> None:
  # bold text with red background
  print(f"\033[1;41m{msg}\033[0m")

def print_hand(player) -> str:
  print("" + ", ".join(f"{card}" for idx, card in enumerate(player.hand)))

def print_board(board, players) -> None:
  top = [cell_str(i, board, players) for i in range(0, 17)][::-1]
  left = [cell_str(i, board, players) for i in range(17, 32)]
  bottom = [cell_str(i, board, players) for i in range(32, 49)]
  right = [cell_str(i, board, players) for i in range(49, 64)][::-1]
                                                  
  print(" ".join(top))
  for i in range(15):
    l = left[i]
    r = right[i]
    print(f"{l}{middle_row(i, top)}{r}")
  print(" ".join(bottom))

def middle_row(i, top):
  c = " " * (len(top)*2-3)
  if i in (0,1,2,3):
    m = " " * (len(top)*2 - 7-i*4)
    s = (2*i+1)*" "
    c = f"{s}❂{m}❂{s}"
  elif i in (11,12,13,14):
    s = (2*(14 - i) +1)*" "
    m = " " * (len(top)*2 - 7-(14-i)*4)
    c = f"{s}❂{m}❂{s}"
  elif i == 7:
      c = "              DOG              "
  return c

def cell_str(i, b, p):
  marble_str = ["❶", "❷", "❸", "❹"]
  if b.track[i] is None:
    if b.start_fields.get(i) is not None:
      return p[b.start_fields[i]].color.value+"◯\033[0m"
    return "."
  else:
    player = p.index(next(pl for pl in p if pl.color == b.track[i].color))
    marble_index = p[player].get_marble(b.track[i])
    return p[player].color.value+marble_str[marble_index-1]+"\033[0m"

def prompter(prompt: str, options: Sequence[str]) -> int:
  while True:
    print(prompt)
    for idx, option in enumerate(options):
      print(f"  {idx + 1}: {option}")
    choice = input("Select an option: ")
    if choice.isdigit():
      choice_idx = int(choice) - 1
      if 0 <= choice_idx < len(options):
        return choice_idx
    error_print("Invalid choice. Please try again.")

def moves_prompter(prompt: str, options: Sequence[str], player: "Player") -> int:
  while True:
    print(prompt)
    for idx, option in enumerate(options):
      marble_str = f"Marble {player.get_marble(option.marble)}"
      if option.marble is None:
          marble_str = "New Marble"
      steps_str = f"Steps {option.steps}" if option.steps is not None else ""
      print(f"  {idx + 1}: {option.card} - {option.kind.name} - {marble_str} - {steps_str}")
    choice = input("Select a move: ")
    if choice.isdigit():
      choice_idx = int(choice) - 1
      if 0 <= choice_idx < len(options):
        return options[choice_idx]
    error_print("Invalid move. Please try again.")