import random
from enum import Enum, auto
import Cards
import Player
import Board

class Colors(Enum):
    RED: str = "\033[91m"
    GREEN: str = "\033[92m"
    YELLOW: str = "\033[93m"
    BLUE: str = "\033[94m"

class Marble:
    def __init__(self, color: Colors):
        self.color = color

    def __str__(self):
        return f"Marble({self.color})"

class Game:
    def __init__(self, players: int):
        self.players = [
            Player.Player(f"Player {color.name.capitalize()}",
            [Marble(color) for _ in range(players)])
            for color in Colors]
        self.deck = Cards.Deck()
        self.deck.shuffle()
        self.draw_num = 6
        self.player_turn = self.players[0]
        self.is_won = False
        self.board = Board.Board(self.players)

    def start_game(self):
        self.board.render()
        self.deal_cards()

    def turn(self):
        print(f"\n{self.player_turn.get_color()}{self.player_turn.name}'s turn:\033[0m")
        print(self.player_turn.print_hand())
        moves = self.legal_cards(self.player_turn)
        for idx, (move_func, card) in enumerate(moves.items(), start=1):
            print(f"Press {idx}: Play {card}")
        if not moves:
            print("No legal moves available. Hand folded.")
            input("Press Enter to end turn...")
            self.player_turn.fold_hand()
            self.player_turn = self.players[(self.players.index(self.player_turn) + 1) % len(self.players)]
            return
        choice = int(input("Select your card to use: ")) - 1
        list(moves.keys())[choice](list(moves.values())[choice])
        input("Press Enter to end turn...")
        self.player_turn = self.players[(self.players.index(self.player_turn) + 1) % len(self.players)]
        self.board.render()
        self.check_winner()

    def deal_cards(self):
        for player in self.players:
            player.receive_hand(self.deck.deal(self.draw_num))
        self.draw_num -= 1
    
    def legal_cards(self, player):
        card_moves = {}
        for card in player.hand:
          if card.rank == Cards.CardType.KING:
              card_moves[self.king_moves] = card
          elif card.rank == Cards.CardType.ACE:
              card_moves[self.ace_moves] = card
          elif card.rank == Cards.CardType.QUEEN:
              if self.board.player_can_move(player, 12):
                  card_moves[self.queen_moves] = card
          elif card.rank == Cards.CardType.TEN:
              if self.board.player_can_move(player, 10):
                  card_moves[self.ten_moves] = card
          elif card.rank == Cards.CardType.NINE:
              if self.board.player_can_move(player, 9):
                  card_moves[self.nine_moves] = card
        return card_moves
    
    def queen_moves(self, card):
        m = self.board.get_movable_marbles(self.player_turn, 12)
        movable_list = ", ".join(f"Marble {i+1}" for i, _ in enumerate(m))
        print(f"Movable marbles: {movable_list}. Press the index of the marble to move.")
        choice = int(input("Select your marble to move: "))
        self.board.move_marble(self.player_turn, 12, m[choice - 1])
    
    def ten_moves(self, card):
        m = self.board.get_movable_marbles(self.player_turn, 10)
        movable_list = ", ".join(f"Marble {i+1}" for i, _ in enumerate(m))
        print(f"Movable marbles: {movable_list}. Press the index of the marble to move.")
        choice = int(input("Select your marble to move: "))
        self.board.move_marble(self.player_turn, 10, m[choice - 1])
    
    def nine_moves(self, card):
        self.board.move_marble(self.player_turn, 9)

    def king_moves(self, card):
        print('Press 1: Move 13 spaces\nPress 2: Start marble')
        choice = int(input("Select your move: ")) - 1
        if choice == 0:
            m = self.board.get_movable_marbles(self.player_turn, 13)
            movable_list = ", ".join(f"Marble {i+1}" for i, _ in enumerate(m))
            print(f"Movable marbles: {movable_list}. Press the index of the marble to move.")
            choice = int(input("Select your marble to move: "))
            self.board.move_marble(self.player_turn, 13, m[choice - 1])
        elif choice == 1:
            self.board.start_marble(self.player_turn)
            self.player_turn.play_card(card)

    def ace_moves(self, card):
        print('Press 1: Move 1 space\nPress 2: Move 11 spaces\nPress 3: Start marble')
        choice = int(input("Select your move: ")) - 1
        if choice == 0:
            m = self.board.get_movable_marbles(self.player_turn, 1)
            movable_list = ", ".join(f"Marble {i+1}" for i, _ in enumerate(m))
            print(f"Movable marbles: {movable_list}. Press the index of the marble to move.")
            choice = int(input("Select your marble to move: "))
            self.board.move_marble(self.player_turn, 1, m[choice - 1])
        elif choice == 1:
            m = self.board.get_movable_marbles(self.player_turn, 11)
            movable_list = ", ".join(f"Marble {i+1}" for i, _ in enumerate(m))
            print(f"Movable marbles: {movable_list}. Press the index of the marble to move.")
            choice = int(input("Select your marble to move: "))
            self.board.move_marble(self.player_turn, 11, m[choice - 1])
        elif choice == 2:
            self.board.start_marble(self.player_turn)
            self.player_turn.play_card(card)

    def joker_moves(self):
        print('Joker')
    
    def check_winner(self):
        for player in self.players:
            if all(marble in player.marbles_in_play for marble in player.marbles):
                self.is_won = True
                print(f"{player.name} has won the game!")
                return player
        return None
        


if __name__ == "__main__":
    game = Game(players=4)
    game.start_game()

    while not game.is_won:
        game.turn()
