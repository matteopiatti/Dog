from enum import Enum, auto

class CardType(Enum):
  TWO = "2"
  THREE = "3"
  FOUR = "4"
  FIVE = "5"
  SIX = "6"
  SEVEN = "7"
  EIGHT = "8"
  NINE = "9"
  TEN = "10"
  JACK = "J"
  QUEEN = "Q"
  KING = "K"
  ACE = "A"
  JOKER = "JO"

class CardSuit(Enum):
  HEARTS = "♥"
  DIAMONDS = "♦"
  CLUBS = "♣"
  SPADES = "♠"
  NONE = "🃟"

class Colors(Enum):
  RED: str = "\033[91m"
  GREEN: str = "\033[92m"
  YELLOW: str = "\033[93m"
  BLUE: str = "\033[94m"