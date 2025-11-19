from .enums import GamePhase, MoveKind
from .rules import marble_allowed_steps
from .cli import select_split_action, render


def start_action(state, action):
    if action.kind == MoveKind.START:
        start_marble(state)
    elif action.kind == MoveKind.MOVE:
        move_marble(state, action.marble, action.steps)
    elif action.kind == MoveKind.SWAP:
        swap_marbles(state, action.marble, action.swap_marble)
    elif action.kind == MoveKind.SPLIT:
        state.phase = GamePhase.SPLIT
        state.split_steps_remaining = 7
        state.split_card = action.card


def swap_marbles(state, marble1, marble2):
    pos1 = state.board.pos_of_marble(marble1)
    pos2 = state.board.pos_of_marble(marble2)

    # print(f"Swapping marbles at positions {pos1} and {pos2}")
    # input("Press Enter to continue...")

    state.board.track[pos1], state.board.track[pos2] = (
        state.board.track[pos2],
        state.board.track[pos1],
    )


def start_marble(state):
    startfield = state.board.start_fields[state.current_player]
    marble = state.board.get_free_player_marble(state.current_player)
    if marble and startfield not in state.board.blocked_fields:
        state.board.track[startfield] = (marble, state.current_player)
        state.board.blocked_fields.add(startfield)
        return True


def move_marble(state, marble, steps):
    player = state.current_player
    if state.player_finished(state.current_player):
        player = state.teammate(state.current_player)
    if marble in state.board.home[player]:
        home_pos = state.board.pos_of_home_marble(marble, player)
        new_pos = home_pos + steps
        state.board.home[player][home_pos] = None
        state.board.home[player][new_pos] = marble
        return True
    marble_pos = state.board.pos_of_marble(marble)
    if marble_pos is None:
        return False
    new_pos = (marble_pos + steps) % state.board.NUM_FIELDS
    startfield = state.board.start_fields[player]

    if state.board.marble_can_move_home(marble, player, steps):
        state.board.track[marble_pos] = (None, None)
        home_slots = state.board.home[player]
        distance_to_start = (startfield - marble_pos) % state.board.NUM_FIELDS
        home_slots[steps - distance_to_start - 1] = marble
        if state.phase == GamePhase.SPLIT:
            move_split(state, marble_pos, distance_to_start)
        return True

    if marble_pos in state.board.start_fields.values():
        state.board.blocked_fields.discard(marble_pos)

    if state.phase == GamePhase.SPLIT:
        move_split(state, marble_pos, steps)

    state.board.track[marble_pos] = (None, None)
    state.board.track[new_pos] = (marble, player)
    return True


def move_split(state, pos, steps):
    move_range = [(pos + i) % state.board.NUM_FIELDS for i in range(1, steps + 1)]
    for p in move_range:
        state.board.track[p] = (None, None)
