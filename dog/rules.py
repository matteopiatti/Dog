from .move import Action
from .enums import MoveKind

def legal_actions(state):
    mip = state.board.player_marbles_in_play(state, state.current_player)

    for card in state.current_player.hand:
      if MoveKind.START in card.kinds and state.board.player_has_startable_marble(state.current_player):
        state.cp_actions.append(Action(
            player=state.current_player,
            card=card,
            kind=MoveKind.START,
            marble=None,
            steps=None,
        ))
      elif MoveKind.MOVE in card.kinds:
         for marble in mip:
            for steps in card.steps:
              if is_valid_move(state, marble, steps):
                state.cp_actions.append(Action(
                    player=state.current_player,
                    card=card,
                    kind=MoveKind.MOVE,
                    marble=marble,
                    steps=steps,
                ))


def is_valid_move(state, marble, step):
  if all(m != marble for m, _ in state.board.track):
      return False
  current_pos = next(i for i, (m, p) in enumerate(state.board.track) if m is marble)
  for s in range(1, abs(step)+1):
      intermediate_pos = (current_pos + (s if step > 0 else -s)) % state.board.NUM_FIELDS
      if intermediate_pos in state.board.occupied_fields:
          return False
  return True

    # mip = state.board.player_marbles_in_play(state, state.current_player)
    # for card in state.current_player.hand:
    #   if card.
    #     for marble in state.current_player.marbles:
    #         if marble not in state.board.track:
    #            start_field = state.board.start_fields[state.players.index(state.current_player)]
    #           if start_field not in state.board.occupied_fields:
    #               actions.append(Action(state.current_player)
    #               break

    #         elif marble in state.board.track:
    #             ## marble is in play
    #     for marble in mip:
    #         if marble not in state.board.track:
    #             start_field = state.board.start_fields[state.players.index(state.current_player)]
    #             if start_field not in state.board.occupied_fields:
    #                 actions.append((card, None))
    #                 break
    #     for marble in mip:
    #         if marble not in state.board.track:
    #             continue
    #         current_pos = next(i for i, m in enumerate(state.board.track) if m is marble)
    #         for s in range(1, abs(card.value)+1):
    #             intermediate_pos = (current_pos + (s if card.value > 0 else -s)) % state.board.NUM_FIELDS
    #             if intermediate_pos in state.board.occupied_fields:
    #                 break
    #         else:
    #             actions.append((card, marble))