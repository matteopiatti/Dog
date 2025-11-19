# action_features.py (for example)

import numpy as np
from dog.enums import CardType, MoveKind, GamePhase
from DogGame import (
    OFFSET_SWITCH,
    OFFSET_PLAY,
    OFFSET_SPLIT,
)

# DogState is your OpenSpiel state wrapper
from DogGame import DogState

RANKS = list(CardType)
KINDS = list(MoveKind)
NUM_RANKS = len(RANKS)
NUM_KINDS = len(KINDS)
NUM_MARBLES = 4

# card_rank one-hot + move_kind one-hot + marble_index one-hot + steps
ACT_DIM = NUM_RANKS + NUM_KINDS + NUM_MARBLES + 1


def encode_action_features(
    dog_state: DogState, player_id: int, action_id: int
) -> np.ndarray:
    """
    Build a fixed-size feature vector for a given action_id in the given DogState.
    Only uses information that is already in the state (no lookahead).
    """
    inner = dog_state._inner
    phase = inner.phase
    current_player = inner.current_player

    # Initialize feature vector
    feat = np.zeros(ACT_DIM, dtype=np.float32)
    offset = 0

    card = None
    move_kind = None
    marble = None
    steps = None

    if phase == GamePhase.SWITCH:
        # SWITCH: select a card index from current player's hand.
        idx = action_id - OFFSET_SWITCH
        card = current_player.hand[idx]
        # we treat this as "switch" with no MoveKind (leave move_kind=None)

    elif phase == GamePhase.PLAY:
        # PLAY: Action objects cached in DogState.legal_actions
        # Make sure cache is populated:
        _ = dog_state.legal_actions(player_id)
        acts = getattr(dog_state, "_cached_actions", [])
        idx = action_id - OFFSET_PLAY
        a = acts[idx]
        card = a.card
        move_kind = a.kind
        marble = a.marble
        steps = a.steps

    elif phase == GamePhase.SPLIT:
        # SPLIT: (marble, step) pairs cached in DogState._split_options
        _ = dog_state.legal_actions(player_id)
        opts = getattr(dog_state, "_split_options", [])
        idx = action_id - OFFSET_SPLIT
        marble, steps = opts[idx]
        # use MoveKind.SPLIT as the kind
        move_kind = MoveKind.SPLIT

    # ---- card rank one-hot ----
    if card is not None:
        r_idx = RANKS.index(card.rank)
        feat[offset + r_idx] = 1.0
    offset += NUM_RANKS

    # ---- move kind one-hot ----
    if move_kind is not None:
        k_idx = KINDS.index(move_kind)
        feat[offset + k_idx] = 1.0
    offset += NUM_KINDS

    # ---- marble index one-hot ----
    # we index marbles relative to current_player
    if marble is not None:
        try:
            m_idx = current_player.marbles.index(marble)
            if 0 <= m_idx < NUM_MARBLES:
                feat[offset + m_idx] = 1.0
        except ValueError:
            # marble not found; leave zeros
            pass
    offset += NUM_MARBLES

    # ---- steps (normalized) ----
    # steps can be negative (4 backwards, etc.); normalize by 13
    if steps is not None:
        feat[offset] = float(steps) / 13.0

    return feat
