from dataclasses import dataclass
from typing import List
from agent.agent import Agent
from dog.enums import CardType, GamePhase, MoveKind
import random

from dog.move import start_action


@dataclass
class TrainingAgent(Agent):
    a_w: List[float]
    s_w: List[float]

    def select_switch_card(self, state, hand):
        return random.choice(hand)

    def select_action(self, state, actions, player):
        best_action = actions[0]
        best_score = -float("inf")
        for a in actions:
            if a.kind == MoveKind.SPLIT:
                continue  # skip split actions for now
            f = self._features_for_action(state, a, player)
            s = sum(wi * fi for wi, fi in zip(self.a_w, f))
            if s > best_score:
                best_score = s
                best_action = a
        return best_action

    def select_split_action(self, action, allowed_steps: dict, state):
        marble = random.choice(list(allowed_steps.keys()))
        steps = random.choice(allowed_steps[marble])
        return marble, steps

    def no_actions(self):
        return None

    def _features_for_action(self, state, action, player):
        tmp_state = state.clone()
        tmp_player = tmp_state.current_player

        # simulate action
        start_action(tmp_state, action)

        # bias term
        f0 = 1.0

        # nr of finished marbles after action
        f1 = tmp_state.board.player_finished_marbles(
            tmp_player
        ) - state.board.player_finished_marbles(player)

        # nr of team finished marbles after action
        f2 = tmp_state.board.team_finished_marbles(
            (tmp_state.teammate(tmp_player), tmp_player)
        ) - state.board.team_finished_marbles((state.teammate(player), player))

        # nr of marbles on board after action
        f3 = len(tmp_state.board.player_marbles_in_play(tmp_player)) - len(
            state.board.player_marbles_in_play(player)
        )

        # nr of marbles in home after action
        f4 = sum(1 for m in tmp_state.board.home[tmp_player] if m is not None) - sum(
            1 for m in state.board.home[player] if m is not None
        )

        # total distance to home after action
        f5 = tmp_state.board.total_distance_to_home(
            tmp_player
        ) - state.board.total_distance_to_home(player)

        # if captured opponent marble
        f6 = self._did_capture(state, state.board, tmp_state.board, player)

        # if teammate total distance to home changed
        f7 = tmp_state.board.total_distance_to_home(
            tmp_state.teammate(tmp_player)
        ) - state.board.total_distance_to_home(state.teammate(player))

        # if oponnent total distance to home changed
        opponents = [
            p for p in state.players if p != player and p != state.teammate(player)
        ]
        f8 = 0
        return [f0, f1, f2, f3, f4, f5, f6, f7, f8]

    def _did_capture(self, state, board_before, board_after, player) -> bool:
        teammate = state.teammate(player)

        def enemy_pos(b):
            return {
                i
                for i, (m, p) in enumerate(b.track)
                if m is not None and p is not player and p is not teammate
            }

        before = enemy_pos(board_before)
        after = enemy_pos(board_after)
        return len(before) > len(after)
