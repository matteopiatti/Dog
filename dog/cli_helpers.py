from dog.enums import CardSuit

def get_printable_card(card):
    pre = ""
    if card.suit in (CardSuit.HEARTS, CardSuit.DIAMONDS):
      pre = "\033[91m"
    elif card.suit in (CardSuit.CLUBS, CardSuit.SPADES):
      pre = "\033[94m"
    elif card.suit == CardSuit.NONE:
      pre = "\033[93m"
    return f"{pre}{card.rank.value}{card.suit.value}\033[0m"

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

def cell_str(state, i):
  if state.board.track[i] == (None, None):
    if i in state.board.start_fields.values():
      color = next(pl.color.value for pl, sf in state.board.start_fields.items() if sf == i)
      return color + "◯" + "\033[0m"
    return "."
  else:
    marble, player = state.board.track[i]
    marble_str = ["❶", "❷", "❸", "❹"][player.marbles.index(marble)]
    return marble.color.value + marble_str + "\033[0m"
  
def print_action_line(idx,card, marble, steps):
  print(f"\033[{idx+2};50H  {idx + 1}: {card} - {marble} - {steps}")