# query_dqn_state.py

import os
import numpy as np

from DogGame import (
    DogGame,
    NUM_ACTIONS,
    OFFSET_SWITCH,
    OFFSET_PLAY,
    OFFSET_SPLIT,
    NUM_SWITCH_ACTIONS,
)
from dog.cli import render
from dqn_agent import DQNAgent

from dog.enums import Colors, CardType, CardSuit, GamePhase, MoveKind
from dog.objects import Player, Marble
from dog.board import Board
from dog.cards import Card, Deck
from dog.state import GameState as DogGameState
from dog.rules import legal_actions
from dog.engine_rl import auto_advance_to_decision
from save import load_state, save_state


# ---------- helpers ----------

COLOR_ORDER = [Colors.RED, Colors.GREEN, Colors.YELLOW, Colors.BLUE]


def parse_card(code: str) -> Card:
    """Parse a card like 'AS', '10H', '7♦', 'JO'."""
    code = code.strip().upper()
    rank_map = {
        "2": CardType.TWO,
        "3": CardType.THREE,
        "4": CardType.FOUR,
        "5": CardType.FIVE,
        "6": CardType.SIX,
        "7": CardType.SEVEN,
        "8": CardType.EIGHT,
        "9": CardType.NINE,
        "10": CardType.TEN,
        "J": CardType.JACK,
        "Q": CardType.QUEEN,
        "K": CardType.KING,
        "A": CardType.ACE,
        "JO": CardType.JOKER,
    }
    suit_map = {
        "H": CardSuit.HEARTS,
        "D": CardSuit.DIAMONDS,
        "C": CardSuit.CLUBS,
        "S": CardSuit.SPADES,
        "O": CardSuit.NONE,  # joker
    }

    if code == "JO":
        return Card(CardType.JOKER, CardSuit.NONE)

    # detect rank part and suit part
    if code.startswith("10"):
        rank_str = "10"
        suit_str = code[2:]
    else:
        rank_str = code[0]
        suit_str = code[1:]

    if rank_str not in rank_map or suit_str not in suit_map:
        raise ValueError(f"Invalid card code: {code}")

    return Card(rank_map[rank_str], suit_map[suit_str])


def build_manual_state():
    """Interactively build a DogGameState."""
    # create players with fixed colors in order
    players = []
    for i, color in enumerate(COLOR_ORDER):
        marbles = [Marble(color) for _ in range(4)]
        players.append(Player(name=f"Player {color.name.title()}", marbles=marbles))

    board = Board(players=players)

    print("=== Set marble positions ===")
    print(
        "For each marble (1..4) of each player, type:\n"
        "  - 'off'   for not in play\n"
        "  - 'tN'    for track index N (0-63)\n"
        "  - 'hN'    for home index N (0-3)\n"
        "Example: t0, t15, h2, off"
    )

    # clear track/home
    board.track = [(None, None)] * board.NUM_FIELDS
    for p in players:
        board.home[p] = [None] * 4
    board.blocked_fields.clear()

    for p in players:
        print(f"\nPlayer {p.name} ({p.color.name}):")
        for mi, m in enumerate(p.marbles):
            while True:
                s = input(f"  Marble {mi+1} position [off|tN|hN]: ").strip().lower()
                if s == "off":
                    # nothing to place
                    break
                elif s.startswith("t"):
                    try:
                        idx = int(s[1:])
                        assert 0 <= idx < board.NUM_FIELDS
                    except Exception:
                        print("    Invalid track index, try again.")
                        continue
                    if board.track[idx] != (None, None):
                        print("    Track cell already occupied, try again.")
                        continue
                    board.track[idx] = (m, p)
                    # if this is a start field, mark blocked
                    if idx in board.start_fields.values():
                        board.blocked_fields.add(idx)
                    break
                elif s.startswith("h"):
                    try:
                        idx = int(s[1:])
                        assert 0 <= idx < 4
                    except Exception:
                        print("    Invalid home index, try again.")
                        continue
                    if board.home[p][idx] is not None:
                        print("    Home slot already occupied, try again.")
                        continue
                    board.home[p][idx] = m
                    break
                else:
                    print("    Invalid format, try again.")

    print("\n=== Set hands ===")
    print(
        "For each player, enter card codes separated by spaces.\n"
        "Examples: 'AS 7D JO', '10H QC'\n"
        "Use O suit for jokers if needed (JO is enough)."
    )

    for p in players:
        while True:
            s = input(f"Hand for {p.name}: ").strip()
            if not s:
                p.hand = []
                break
            try:
                codes = s.split()
                cards = [parse_card(c) for c in codes]
                p.hand = cards
                break
            except Exception as e:
                print(f"    Error: {e}. Try again.")

    # choose current player
    print("\n=== Current player ===")
    for i, p in enumerate(players):
        print(f"  {i}: {p.name}")
    while True:
        s = input("Index of current player [0-3]: ").strip()
        try:
            idx = int(s)
            assert 0 <= idx < len(players)
            current_player = players[idx]
            break
        except Exception:
            print("    Invalid index, try again.")

    # basic teams as in engine
    teams = [(players[0], players[2]), (players[1], players[3])]

    # dummy deck (empty) and discard
    deck = Deck(cards=[])
    discard = []

    # construct inner GameState
    inner = DogGameState(
        players=players,
        board=board,
        deck=deck,
        discard_pile=discard,
        draw_size=0,
        current_player=current_player,
        last_started_player=current_player,
        finished=False,
        winner=None,
        phase=GamePhase.TURN,
        cp_actions=[],
        teams=teams,
        num_rounds=0,
        agents={},  # not used here
    )

    # advance to decision phase (fills cp_actions / phase)
    render(inner)
    save_state(inner, "saved_state.json")
    auto_advance_to_decision(inner)

    return inner


def encode_state_to_obs_and_actions(game: DogGame, inner: DogGameState, player_id: int):
    """Wrap inner GameState in DogState, get obs + legal action ids and cached actions."""
    # Build DogState via game.make_copy or constructor
    from DogGame import DogState  # import here to avoid circulars

    state = DogState(game, inner)
    legal = state.legal_actions(player_id)  # populates _cached_actions/_split_options
    obs = state.observation_tensor(player_id)
    return state, obs, legal


def describe_action_from_id(state, action_id: int, player_id: int):
    """Decode and describe an action id as human-readable text."""
    phase = state._inner.phase

    if phase == GamePhase.SWITCH:
        idx = action_id - OFFSET_SWITCH
        player = state._inner.current_player
        card = player.hand[idx]
        return f"SWITCH: give card {card.rank.name} of {card.suit.name} to teammate"

    if phase == GamePhase.PLAY:
        idx = action_id - OFFSET_PLAY
        acts = getattr(state, "_cached_actions", None)
        if acts is None or idx >= len(acts):
            return f"PLAY: action id {action_id} (index {idx})"
        a = acts[idx]
        card = a.card
        txt = f"PLAY: use {card.rank.name} of {card.suit.name} as {a.kind.name}"
        if a.kind == MoveKind.MOVE:
            m_idx = a.player.marbles.index(a.marble)
            txt += f", move marble #{m_idx+1} of {a.player.name} by {a.steps} steps"
        elif a.kind == MoveKind.START:
            txt += f", start a marble for {a.player.name}"
        elif a.kind == MoveKind.SWAP:
            m1_idx = a.player.marbles.index(a.marble)
            m2_idx = a.swap_player.marbles.index(a.swap_marble)
            txt += (
                f", swap marble #{m1_idx+1} of {a.player.name} with "
                f"marble #{m2_idx+1} of {a.swap_player.name}"
            )
        elif a.kind == MoveKind.SPLIT:
            txt += " (SPLIT card: follow-up split submoves required)"
        return txt

    if phase == GamePhase.SPLIT:
        idx = action_id - OFFSET_SPLIT
        opts = getattr(state, "_split_options", None)
        if opts is None or idx >= len(opts):
            return f"SPLIT: action id {action_id} (index {idx})"
        marble, steps = opts[idx]
        player = state._inner.current_player
        m_idx = player.marbles.index(marble)
        return f"SPLIT: move marble #{m_idx+1} of {player.name} by {steps} steps"

    return f"UNKNOWN PHASE {phase.name}, action id {action_id}"


# ---------- main ----------


def main():
    model_path = "dog_dqn.pt"
    if not os.path.exists(model_path):
        print(f"Model file '{model_path}' not found.")
        return

    game = DogGame()

    # dummy env to infer obs size (not strictly needed if you know it)
    from open_spiel.python import rl_environment

    env = rl_environment.Environment(game=game)
    ts = env.reset()
    obs_dim = len(ts.observations["info_state"][0])

    dqn = DQNAgent(
        player_id=0,
        obs_dim=obs_dim,
        num_actions=NUM_ACTIONS,
    )
    dqn.load(model_path)
    print(f"Loaded DQN model from {model_path}")

    print("\n=== Build a custom state to query ===")
    print("Want to load state? (y/n): ", end="")
    choice = input().strip().lower()
    if choice == "y":
        print("Loading state from 'saved_state.json'...")
        inner = load_state("saved_state.json")
    else:
        print("Building manual state...")
        inner = build_manual_state()

    # assume we always query from player 0's POV
    player_id = 0
    state, obs, legal = encode_state_to_obs_and_actions(game, inner, player_id)
    print(inner)

    if not legal:
        print("No legal actions for this player in this state.")
        return

    print(f"\nLegal action ids: {legal}")

    # get best action from DQN (greedy)
    action_id = dqn.select_action(
        np.array(obs, dtype=np.float32),
        legal_actions=legal,
        eval_mode=True,
    )

    desc = describe_action_from_id(state, action_id, player_id)

    print("\n=== DQN suggestion ===")
    print(f"Chosen action id: {action_id}")
    print(f"Interpretation: {desc}")


if __name__ == "__main__":
    main()
