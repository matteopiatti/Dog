from .state import GameState
from .board import Board
from .cards import Deck
from .objects import Player, Marble
from .enums import Colors, GamePhase
from .cli import print_no_actions, select_action, select_switch_card
from .rules import legal_actions
from .move import start_action


def setup_game(num_players: int) -> GameState:
    deck = Deck()
    deck.shuffle()
    players = [
        Player(
            name=f"Player {color.name.capitalize()}",
            marbles=[Marble(color) for _ in range(4)],
        )
        for color in list(Colors)[:num_players]
    ]
    board = Board(players=players)
    teams = [
        (players[0], players[2]),
        (players[1], players[3]),
    ]
    return GameState(
        players=players,
        board=board,
        deck=deck,
        current_player=players[0],
        last_started_player=players[0],
        teams=teams,
    )


def step(state: GameState) -> GameState:
    state.check_winner()
    if state.finished:
        return state

    if state.phase == GamePhase.DEAL:
        state.advance_round()
        deal_cards(state)
        state.phase = GamePhase.SWITCH
        return state

    elif state.phase == GamePhase.SWITCH:
        selected_card = select_switch_card(state)
        state.switch_cards.append((state.current_player, selected_card))

        if len(state.switch_cards) < len(state.players):
            state.advance_player()
            return state
        elif len(state.switch_cards) == len(state.players):
            for player_from, card in state.switch_cards:
                state.teammate(player_from).receive_card(card)
                player_from.play_card(card)
            state.switch_cards.clear()
            state.phase = GamePhase.TURN
            return state
    elif state.phase == GamePhase.TURN:
        if state.empty_hands:
            state.phase = GamePhase.DEAL
            return state
        state.cp_actions = legal_actions(state)
        if not state.cp_actions:
            print_no_actions()
            state.current_player.fold()
            state.advance_player()
        else:
            state.phase = GamePhase.PLAY
        return state

    elif state.phase == GamePhase.PLAY:
        action = select_action(state)
        start_action(state, action)
        state.current_player.play_card(action.card)
        state.reset_actions()
        state.advance_player()
        state.phase = GamePhase.TURN
        return state


def deal_cards(state: GameState) -> None:
    for player in state.players:
        player.receive_hand(state.deck.deal(state.draw_size))
