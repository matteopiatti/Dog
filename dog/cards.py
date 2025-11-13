from .enums import CardType, CardSuit, MoveKind
import random

class Card:
  def __init__(self, rank: CardType, suit: CardSuit, kinds: list[MoveKind]):
    self.rank = rank
    self.suit = suit
    self.kinds = kinds
    self.steps = []

class Deck:
  def __init__(self, cards=None):
    if cards is None:
        suits = [s for s in CardSuit if s != CardSuit.NONE]
        ranks = [r for r in CardType if r != CardType.JOKER]
        base = [Card(r, s, MoveKind.MOVE) for s in suits for r in ranks]
        jokers = [Card(CardType.JOKER, CardSuit.NONE, MoveKind.MOVE)] * 3
        self.cards = (base + jokers) * 2
        for card in self.cards:
           give_card_move_kind(card)
           give_card_steps(card)
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


def give_card_move_kind(card: Card) -> None:
    if card.rank in [CardType.ACE, CardType.KING]:
        card.kinds = [MoveKind.START, MoveKind.MOVE]
    elif card.rank == CardType.JOKER:
        card.kinds = [MoveKind.START, MoveKind.MOVE, MoveKind.SPLIT, MoveKind.SWAP]
    elif card.rank == CardType.SEVEN:
        card.kinds = [MoveKind.SPLIT]
    elif card.rank == CardType.JACK:
        card.kinds = [MoveKind.SWAP]
    else:
        card.kinds = [MoveKind.MOVE]

def give_card_steps(card: Card) -> None:
    if card.rank == CardType.ACE:
        card.steps = [1, 11]
    elif card.rank == CardType.KING:
        card.steps = [13]
    elif card.rank == CardType.QUEEN:
        card.steps = [12]
    elif card.rank == CardType.TEN:
        card.steps = [10]
    elif card.rank == CardType.NINE:
        card.steps = [9]
    elif card.rank == CardType.EIGHT:
        card.steps = [8]
    elif card.rank == CardType.SEVEN:
        card.steps = [1, 2, 3, 4, 5, 6, 7]
    elif card.rank == CardType.SIX:
        card.steps = [6]
    elif card.rank == CardType.FIVE:
        card.steps = [5]
    elif card.rank == CardType.FOUR:
        card.steps = [4, -4]
    elif card.rank == CardType.THREE:
        card.steps = [3]
    elif card.rank == CardType.TWO:
        card.steps = [2]
    elif card.rank == CardType.JOKER:
        card.steps = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]