from .action import Action
from .enums import MoveKind

def legal_actions(state):
    mip = state.board.player_marbles_in_play(state.current_player)
    mih = state.board.home[state.current_player]

    for card in state.current_player.hand:
      if MoveKind.START in card.kinds and state.board.player_has_startable_marble(state.current_player):
        state.cp_actions.append(Action(
            player=state.current_player,
            card=card,
            kind=MoveKind.START,
            marble=None,
            swap_marble=None,
            swap_player=None,
            steps=None,
        ))
      if MoveKind.MOVE in card.kinds:
        for marble in mip:
            for steps in card.steps:
              if is_valid_move(state, marble, steps):
                state.cp_actions.append(Action(
                    player=state.current_player,
                    card=card,
                    kind=MoveKind.MOVE,
                    marble=marble,
                    swap_marble=None,
                    swap_player=None,
                    steps=steps,
                ))
        for marble in mih:
            if marble is not None:
              for steps in card.steps:
                if is_valid_home_move(state, marble, state.current_player, steps):
                  state.cp_actions.append(Action(
                      player=state.current_player,
                      card=card,
                      kind=MoveKind.MOVE,
                      marble=marble,
                      swap_marble=None,
                      swap_player=None,
                      steps=steps,
                  ))
      if MoveKind.SWAP in card.kinds:
        for m in mip:
          for i, (m2, p) in enumerate(state.board.track):
            if p is not state.current_player and m2 is not None:
              if is_valid_swap(state, m, m2):
                state.cp_actions.append(Action(
                    player=state.current_player,
                    card=card,
                    kind=MoveKind.SWAP,
                    marble=m,
                    swap_marble=m2,
                    swap_player=p,
                    steps=None,
                ))
      if MoveKind.SPLIT in card.kinds:
        if is_valid_split(state):
          state.cp_actions.append(Action(
              player=state.current_player,
              card=card,
              kind=MoveKind.SPLIT,
              marble=None,
              swap_marble=None,
              swap_player=None,
              steps=None,
          ))

def is_valid_split(state):
  allowed_steps = marble_allowed_steps(state)
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
  if len(mip) == 1 and is_valid_move(state, mip[0], max_steps):
    return {mip[0]: [max_steps]}
  for m in mip:
    current_pos = next(i for i, (mar, p) in enumerate(state.board.track) if mar is m)
    for step in range(1, max_steps + 1):
        intermediate_pos = (current_pos + step) % state.board.NUM_FIELDS
        if intermediate_pos in state.board.occupied_fields:
            break
        allowed_steps.setdefault(m, []).append(step)
  return allowed_steps

def is_valid_swap(state, m1, m2):
  pos_m1 = next(i for i, (m, p) in enumerate(state.board.track) if m is m1)
  pos_m2 = next(i for i, (m, p) in enumerate(state.board.track) if m is m2)
  if pos_m1 not in state.board.occupied_fields and pos_m2 not in state.board.occupied_fields:
      return True
  return False

def is_valid_move(state, marble, step):
  if all(m != marble for m, _ in state.board.track):
      return False
  current_pos = next(i for i, (m, p) in enumerate(state.board.track) if m is marble)
  for s in range(1, abs(step)+1):
      intermediate_pos = (current_pos + (s if step > 0 else -s)) % state.board.NUM_FIELDS
      if intermediate_pos in state.board.occupied_fields:
          return False
  return True

def is_valid_home_move(state, marble, player, step):
  if all(m != marble for m in state.board.home[player]):
      return False
  current_pos = next(i for i, m in enumerate(state.board.home[player]) if m is marble)
  move_range = range(current_pos, min(current_pos + step, 3))
  print(move_range)
  input("debug")
  for pos in move_range:
      if state.board.home[player][pos] is not None:
          return False
  if current_pos + step >= 4:
      return False
  return True