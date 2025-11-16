from .objects import Action
from .enums import MoveKind


def legal_actions(state):
    actions = []
    player = state.current_player
    if state.player_finished(state.current_player):
        player = state.teammate(state.current_player)
    mip = state.board.player_marbles_in_play(player)
    mih = state.board.home[player]
    can_start = state.board.player_has_startable_marble(player)

    for card in state.current_player.hand:
        if MoveKind.START in card.kinds and can_start:
            actions.append(
                Action(
                    player=player,
                    card=card,
                    kind=MoveKind.START,
                )
            )
        if MoveKind.MOVE in card.kinds:
            for marble in mip:
                for steps in card.steps:
                    if is_valid_move(state, marble, steps):
                        actions.append(
                            Action(
                                player=player,
                                card=card,
                                kind=MoveKind.MOVE,
                                marble=marble,
                                steps=steps,
                            )
                        )
            for marble in mih:
                if marble is not None:
                    for steps in card.steps:
                        if is_valid_home_move(state, marble, player, steps):
                            actions.append(
                                Action(
                                    player=player,
                                    card=card,
                                    kind=MoveKind.MOVE,
                                    marble=marble,
                                    steps=steps,
                                )
                            )
        if MoveKind.SWAP in card.kinds:
            for m in mip:
                for i, (m2, p) in enumerate(state.board.track):
                    if p is not player and m2 is not None:
                        if is_valid_swap(state, m, m2):
                            actions.append(
                                Action(
                                    player=player,
                                    card=card,
                                    kind=MoveKind.SWAP,
                                    marble=m,
                                    swap_marble=m2,
                                    swap_player=p,
                                )
                            )
        if MoveKind.SPLIT in card.kinds:
            if is_valid_split(state):
                actions.append(
                    Action(
                        player=player,
                        card=card,
                        kind=MoveKind.SPLIT,
                    )
                )
    return actions


def is_valid_split(state):
    allowed_steps = marble_allowed_steps(state)
    if any(7 in steps for steps in allowed_steps.values()):
        return True
    for m, steps in allowed_steps.items():
        if 7 in steps:
            return True
    for m1, steps1 in allowed_steps.items():
        for m2, steps2 in allowed_steps.items():
            if m1 is not m2:
                for s1 in steps1:
                    for s2 in steps2:
                        if s1 + s2 == 7:
                            return True
    return False


def marble_allowed_steps(state, max_steps=7):
    allowed_steps = {}
    mip = state.board.player_marbles_in_play(state.current_player)
    mih = state.board.home[state.current_player]
    m_finished = state.board.player_finished_marbles(state.current_player)
    movable_home_marbles = len([m for m in mih if m is not None]) - m_finished
    if (
        len(mip) == 1
        and movable_home_marbles == 0
        and is_valid_move(state, mip[0], max_steps)
    ):
        return {mip[0]: [max_steps]}
    for m in mip:
        marble_pos = state.board.pos_of_marble(m)
        for step in range(1, max_steps + 1):
            intermediate_pos = (marble_pos + step) % state.board.NUM_FIELDS
            if intermediate_pos in state.board.blocked_fields:
                break
            allowed_steps.setdefault(m, []).append(step)
    for m in mih:
        if m is not None:
            marble_pos = state.board.pos_of_home_marble(m, state.current_player)
            for step in range(1, max_steps + 1):
                if is_valid_home_move(state, m, state.current_player, step):
                    allowed_steps.setdefault(m, []).append(step)
    return allowed_steps


def is_valid_swap(state, m1, m2):
    pos_m1 = state.board.pos_of_marble(m1)
    pos_m2 = state.board.pos_of_marble(m2)
    if (
        pos_m1 not in state.board.blocked_fields
        and pos_m2 not in state.board.blocked_fields
    ):
        return True
    return False


def is_valid_move(state, marble, step):
    current_pos = state.board.pos_of_marble(marble)
    if current_pos is None:
        return False
    dir = 1 if step > 0 else -1
    for s in range(1, abs(step) + 1):
        intermediate_pos = (current_pos + dir * s) % state.board.NUM_FIELDS
        if intermediate_pos in state.board.blocked_fields:
            return False
    return True


def is_valid_home_move(state, marble, player, step):
    if step <= 0 or step > 3:
        return False

    home_row = state.board.home[player]
    current_pos = state.board.pos_of_home_marble(marble, player)
    if current_pos is None:
        return False

    target = current_pos + step
    if target >= len(home_row):
        return False

    for pos in range(current_pos + 1, target + 1):
        if home_row[pos] is not None:
            return False

    return True
