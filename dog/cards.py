from .enums import CardType, CardSuit
import random

class Card:
  def __init__(self, rank: CardType, suit: CardSuit):
    self.rank = rank
    self.suit = suit

  def __str__(self):
    pre = ""
    if self.suit in (CardSuit.HEARTS, CardSuit.DIAMONDS):
      pre = "\033[91m"
    elif self.suit in (CardSuit.CLUBS, CardSuit.SPADES):
      pre = "\033[94m"
    elif self.suit == CardSuit.NONE:
      pre = "\033[93m"
    return f"{pre}{self.rank.value}{self.suit.value}\033[0m"

class Deck:
  def __init__(self, cards=None):
    if cards is None:
        suits = [s for s in CardSuit if s != CardSuit.NONE]
        ranks = [r for r in CardType if r != CardType.JOKER]
        base = [Card(r, s) for s in suits for r in ranks]
        jokers = [Card(CardType.JOKER, CardSuit.NONE)] * 3
        self.cards = (base + jokers) * 2
    else:
        self.cards = list(cards)

  def deal(self, num_cards):
    dealt = []
    while num_cards > 0:
        if not self.cards:
            self.__init__()
            self.shuffle()
        take = min(num_cards, len(self.cards))
        dealt += self.cards[:take]
        self.cards = self.cards[take:]
        num_cards -= take
    return dealt


  def shuffle(self):
    random.shuffle(self.cards)