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
        steps = 7
        while steps > 0:
            render(state)
            allowed_steps = marble_allowed_steps(state, steps)
            m, s = select_split_action(action, allowed_steps)
            print(f"Selected marble: {m}, steps: {s}")
            move_marble(state, m, s)
            steps -= s


def swap_marbles(state, marble1, marble2):
    pos1 = next(i for i, (m, p) in enumerate(state.board.track) if m is marble1)
    pos2 = next(i for i, (m, p) in enumerate(state.board.track) if m is marble2)

    state.board.track[pos1], state.board.track[pos2] = (
        state.board.track[pos2],
        state.board.track[pos1],
    )


def start_marble(state):
    startfield = state.board.start_fields[state.current_player]
    marble = state.board.get_free_player_marble(state.current_player)
    if marble and startfield not in state.board.occupied_fields:
        state.board.track[startfield] = (marble, state.current_player)
        state.board.occupied_fields.add(startfield)
        return True


def move_marble(state, marble, steps):
    marble_pos = state.board.pos_of_marble(marble)
    new_pos = (marble_pos + steps) % state.board.NUM_FIELDS
    startfield = state.board.start_fields[state.current_player]

    if state.board.marble_can_move_home(marble, state.current_player, steps):
        state.board.track[marble_pos] = (None, None)
        home_slots = state.board.home[state.current_player]
        distance_to_start = (startfield - marble_pos) % state.board.NUM_FIELDS
        home_slots[steps - distance_to_start - 1] = marble
        if state.phase == GamePhase.SPLIT:
            move_split(state, marble_pos, steps)
        return True

    if marble_pos in state.board.start_fields.values():
        state.board.occupied_fields.discard(marble_pos)

    if state.phase == GamePhase.SPLIT:
        move_split(state, marble_pos, steps)

    state.board.track[marble_pos] = (None, None)
    state.board.track[new_pos] = (marble, state.current_player)
    return True


def move_split(state, pos, steps):
    move_range = [(pos + i) % state.board.NUM_FIELDS for i in range(1, steps + 1)]
    for pos in move_range:
        state.board.track[pos] = (None, None)
