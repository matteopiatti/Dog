from typing import Sequence
from dog.player import Player
from rich import print, box
from rich.console import Console, Group
from rich.panel import Panel
from rich.columns import Columns
from rich.layout import Layout
from rich.text import Text
from rich.table import Table
from rich.prompt import Prompt
from rich.align import Align
from .cli_helpers import ListPrompt

console = Console()

# this should render whatever is the step. It should not contain game logic.
def render(state) -> None:
  board_str = create_board_string(state.board, state.players)
  with console.screen():
    OPTIONS = ["Option 1", "Option 2", "Option 3"]
    h = console.size.height
    target = int(h * 0.99)
    layout = Layout()
    layout.split_row(
      Layout(name="Game"),
      Layout(name="Action", ratio=2),
    )
    layout["Game"].split_column(
      Layout(name="Board", ratio=3),
      Layout(name="Hand"),
    )
    # add text to layout Board

    layout["Game"]["Board"].update(Panel("board_str", title="Board"))
    layout["Action"].update(Panel(Align.center("table", vertical="middle"), title="Action"))
    console.print(layout, height=target)
    input("Press Enter to continue...")
  # clear_screen()
  # print_board(state.board, state.players)
  # for player in state.players:
  #   print_hand(player)

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

def create_board_string(board, players) -> str:
  board_string = ""
  top = [cell_str(i, board, players) for i in range(0, 17)][::-1]
  left = [cell_str(i, board, players) for i in range(17, 32)]
  bottom = [cell_str(i, board, players) for i in range(32, 49)]
  right = [cell_str(i, board, players) for i in range(49, 64)][::-1]
                                                  
  board_string += " ".join(top) + "\n"
  for i in range(15):
    l = left[i]
    r = right[i]
    board_string += f"{l}{middle_row(i, top)}{r}\n"
  board_string += " ".join(bottom)
  return board_string

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

# prompter should also be usable by bots and not contain any game logic.
def select_action(state, options: Sequence[str]) -> int:
  return prompter(state, "Select an action:", options)

def prompter(state, prompt:str, options: Sequence[str]) -> int:
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


def old_prompter(prompt: str, options: Sequence[str]) -> int:
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