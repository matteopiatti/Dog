# dog/engine_rl.py (new file, thin wrapper)
from .engine import Board, Deck
from .state import GameState
from .objects import Player, Marble
from .rules import legal_actions
from .enums import Colors, GamePhase


def setup_game_rl(num_players: int = 4) -> GameState:
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
        agents={},  # not used in RL version
    )


def auto_advance_to_decision(state: GameState) -> None:

    while True:
        state.check_winner()
        if state.finished:
            return

        if state.phase == GamePhase.DEAL:
            state.advance_round()
            deal_cards(state)

            state.phase = GamePhase.TURN
            # SWITCH is a player decision phase -> stop
            # return

        if state.phase == GamePhase.SWITCH:
            # decision by current player which card to give -> stop
            state.phase = GamePhase.TURN
            return

        elif state.phase == GamePhase.TURN:
            if state.empty_hands:
                state.phase = GamePhase.DEAL
                continue  # will deal in next loop

            state.cp_actions = legal_actions(state)
            if not state.cp_actions:
                # equivalent of agent.no_actions + fold + advance_player
                state.current_player.fold()
                state.advance_player()
                # stay in TURN, recompute in next loop
                continue
            else:
                # there are actions, so we are at a decision point
                state.phase = GamePhase.PLAY
                return

        elif state.phase == GamePhase.PLAY:
            # should only happen immediately after selecting an action
            # from OpenSpiel side; we will immediately move and advance.
            # In auto_advance() we should never "land" here.
            return
            # raise RuntimeError("auto_advance_to_decision called in PLAY phase")

        elif state.phase == GamePhase.SPLIT:
            # split decisions are handled as part of the action itself
            # from OpenSpiel's point of view; we won't stop here
            return
        else:
            raise RuntimeError("auto_advance_to_decision called in PLAY phase")


def deal_cards(state: GameState) -> None:
    for player in state.players:
        player.receive_hand(state.deck.deal(state.draw_size))
