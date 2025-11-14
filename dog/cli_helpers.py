from dog.enums import CardSuit, MoveKind

def get_printable_card(card):
    pre = ""
    if card.suit in (CardSuit.HEARTS, CardSuit.DIAMONDS):
      pre = "\033[91m"
    elif card.suit in (CardSuit.CLUBS, CardSuit.SPADES):
      pre = "\033[94m"
    elif card.suit == CardSuit.NONE:
      pre = "\033[93m"
    return f"{pre}{card.rank.value}{card.suit.value}\033[0m"

def middle_row(i, top, state):
  c = " " * (len(top)*2-3)
  if i in (0,1,2,3):
    m = " " * (len(top)*2 - 7-i*4)
    s = (2*i+1)*" "
    home_left = get_home_field(state, state.players[1], i)
    home_right = get_home_field(state, state.players[0], i)
    c = f"{s}{home_left}{m}{home_right}{s}"
  elif i in (11,12,13,14):
    s = (2*(14 - i) +1)*" "
    m = " " * (len(top)*2 - 7-(14-i)*4)
    home_left = get_home_field(state, state.players[2], 14-i)
    home_right = get_home_field(state, state.players[3], 14-i)
    c = f"{s}{home_left}{m}{home_right}{s}"
  elif i == 7:
      c = "              DOG              "
  return c

def get_home_field(state, player, i):
  home = state.board.home[player]
  marble = home[i]
  if marble is not None:
    return marble.color.value + get_printable_marble(marble, player) + "\033[0m"
  return "❂"

def cell_str(state, i):
  if state.board.track[i] == (None, None):
    if i in state.board.start_fields.values():
      color = next(pl.color.value for pl, sf in state.board.start_fields.items() if sf == i)
      return color + "◯" + "\033[0m"
    return "."
  else:
    marble, player = state.board.track[i]
    return marble.color.value + get_printable_marble(marble, player) + "\033[0m"
  
def get_printable_marble(marble, player):
  if marble is None:
      
      return f"{player.color.value}New ●\033[0m"
  marble_str = ["❶", "❷", "❸", "❹"][player.marbles.index(marble)]
  return f"{player.color.value}{marble_str}\033[0m"

def print_action_line(action, idx):
  steps = f"Steps {action.steps}" if action.steps is not None else ""
  marble = get_printable_marble(action.marble, action.player)
  swap_marble = get_printable_marble(action.swap_marble, action.swap_player) if action.swap_marble else ""
  if action.kind == MoveKind.SPLIT:
    marble = "Split"
  card = get_printable_card(action.card)
  print(f"\033[{idx+2};50H  {idx + 1}: {card} - {marble} {swap_marble} {' - ' + steps if steps else ''}\033[0m")

def print_action_selection_line(actions):
  print_string = ""
  for idx, opt in enumerate(actions):
    marble_str = get_printable_marble(opt.marble, opt.player)
    card_str = get_printable_card(opt.card)
    print_string += f"  {idx + 1}: {marble_str if opt.marble is not None else ''} {card_str}"
  return print_string