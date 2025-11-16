from .enums import CardType, CardSuit, MoveKind
import random


class Card:
    def __init__(self, rank: CardType, suit: CardSuit):
        self.rank = rank
        self.suit = suit
        self.kinds = MOVE_KINDS.get(rank, [MoveKind.MOVE])
        self.steps = STEPS.get(rank, [])


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


MOVE_KINDS = {
    CardType.ACE: [MoveKind.START, MoveKind.MOVE],
    CardType.KING: [MoveKind.START, MoveKind.MOVE],
    CardType.JOKER: [MoveKind.START, MoveKind.MOVE, MoveKind.SPLIT, MoveKind.SWAP],
    CardType.SEVEN: [MoveKind.SPLIT],
    CardType.JACK: [MoveKind.SWAP],
}

STEPS = {
    CardType.ACE: [1, 11],
    CardType.KING: [13],
    CardType.QUEEN: [12],
    CardType.TEN: [10],
    CardType.NINE: [9],
    CardType.EIGHT: [8],
    CardType.SEVEN: [1, 2, 3, 4, 5, 6, 7],
    CardType.SIX: [6],
    CardType.FIVE: [5],
    CardType.FOUR: [4, -4],
    CardType.THREE: [3],
    CardType.TWO: [2],
    CardType.JOKER: [1, 2, 3, -4, 4, 5, 6, 8, 9, 10, 11, 12, 13],
}
