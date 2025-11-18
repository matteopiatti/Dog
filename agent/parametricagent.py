from dataclasses import dataclass
from typing import List
from agent.agent import Agent
from dog.enums import CardType, GamePhase, MoveKind
from dog.objects import Action
from dog.rules import legal_actions
from dog.state import GameState as GS
from dog.move import start_action
import random
import copy


@dataclass
class ParametricAgent(Agent):
    w: List[float]

    def select_switch_card(self, state, hand):
        best_card = hand[0]
        best_score = -float("inf")
        for c in hand:
            f = self._features_for_switch(state, c, state.current_player)
            s = sum(wi * fi for wi, fi in zip(self.w, f))
            if s > best_score:
                best_score = s
                best_card = c
        return best_card

    def select_action(self, state, actions, player):
        best_action = actions[0]
        best_score = -float("inf")
        for a in actions:
            if a.kind == MoveKind.SPLIT:
                continue  # skip split actions for now
            f = self._features_for_action(state, a, player)
            s = sum(wi * fi for wi, fi in zip(self.w, f))
            if s > best_score:
                best_score = s
                best_action = a
        return best_action

    def select_split_action(self, action, allowed_steps: dict, state):
        # leave random for now
        marble = random.choice(list(allowed_steps.keys()))
        steps = random.choice(allowed_steps[marble])
        return marble, steps

    def no_actions(self):
        return None

    # ---------- Feature extraction ---------

    def _features_for_action(self, state, action, player):
        board = state.board
        f0 = 1.0  # bias term

        my_finished_before = board.player_finished_marbles(player)
        team_finished_before = board.team_finished_marbles(
            (state.teammate(player), player)
        )
        on_board_before = board.player_marbles_in_play(player)
        home_before = sum(1 for m in board.home[player] if m is not None)
        danger_before = self._total_danger(board, player)
        dist_before = self._total_distance_to_home(board, player)

        new_board = self._simulate_action(state, board, action, player)

        my_finished_after = new_board.player_finished_marbles(player)
        team_finished_after = new_board.team_finished_marbles(
            (state.teammate(player), player)
        )
        on_board_after = new_board.player_marbles_in_play(player)
        home_after = sum(1 for m in new_board.home[player] if m is not None)
        danger_after = self._total_danger(new_board, player)
        dist_after = self._total_distance_to_home(new_board, player)

        f1 = my_finished_after - my_finished_before
        f2 = team_finished_after - team_finished_before
        f3 = len(on_board_after) - len(on_board_before)
        f4 = home_after - home_before
        f5 = 1.0 if self._did_capture(state, board, new_board, player) else 0.0
        f6 = (dist_before - dist_after) / 10.0
        f7 = danger_after
        f8 = danger_before - danger_after
        f9 = (
            1.0
            if self._moved_teammate_marble(
                board, new_board, player, state.teammate(player)
            )
            else 0.0
        )
        return [f0, f1, f2, f3, f4, f5, f6, f7, f8, f9]

    def _features_for_switch(self, state, card, player):
        teammate = state.teammate(player)

        teammate_in_play = len(state.board.player_marbles_in_play(teammate))
        teammate_stuck = 4 - teammate_in_play

        own_in_play = len(state.board.player_marbles_in_play(player))
        own_stuck = 4 - own_in_play

        can_play = self._no_moves_after_giving(state, player, card)

        return [
            1 if MoveKind.SWAP in card.kinds else 0,
            1 if card.rank == CardType.JOKER else 0,
            max(card.steps) if card.steps else 0,
            teammate_in_play,
            teammate_stuck,
            own_in_play,
            own_stuck,
            1 if card.steps and max(card.steps) <= 4 else 0,
            1 if card.steps and max(card.steps) >= 10 else 0,
            can_play,
        ]

    # ---------- Helpers ----------

    def _simulate_action(self, state, board, action, player):
        new_board = state.board.copy()

        tmp_state = GS(
            players=state.players,
            board=new_board,
            deck=state.deck,
            current_player=player,
            last_started_player=state.last_started_player,
            teams=state.teams,
        )
        start_action(tmp_state, action)
        return tmp_state.board

    def _total_danger(self, board, player, K: int = 6) -> int:
        danger = 0
        for m in board.player_marbles_in_play(player):
            pos = board.pos_of_marble(m)
            if pos is None:
                continue
            for d in range(1, K + 1):
                check_pos = (pos - d) % board.NUM_FIELDS
                m2, p2 = board.track[check_pos]
                if m2 is not None and p2 is not player:
                    danger += 1
        return danger

    def _total_distance_to_home(self, board, player) -> int:
        startfield = board.start_fields[player]
        total = 0
        for m in board.player_marbles_in_play(player):
            pos = board.pos_of_marble(m)
            if pos is None:
                continue
            distance = (startfield - pos) % board.NUM_FIELDS
            total += distance
        return total

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

    def _no_moves_after_giving(self, state: GS, player, card) -> int:
        # shallow copy of board (your custom method)
        board_copy = state.board.copy()

        # build a temporary state
        tmp_state = GS(
            players=state.players,
            board=board_copy,
            deck=state.deck,
            current_player=player,
            last_started_player=state.last_started_player,
            teams=state.teams,
        )
        tmp_state.phase = GamePhase.TURN

        # ---- TEMPORARY HAND SWAP ----
        orig_hand = player.hand
        try:
            # shadow copy
            tmp_hand = list(orig_hand)
            if card in tmp_hand:
                tmp_hand.remove(card)
            # assign shadow hand
            player.hand = tmp_hand

            # test move availability
            actions = legal_actions(tmp_state)
            return 1 if not actions else 0

        finally:
            # ---- RESTORE ORIGINAL HAND ----
            player.hand = orig_hand

    def _moved_teammate_marble(self, old_board, new_board, player, teammate):
        def positions(b, pl):
            return {m: b.pos_of_marble(m) for m in b.player_marbles_in_play(pl)}

        before = positions(old_board, teammate)
        after = positions(new_board, teammate)
        # if any teammate marble changed position, we moved it
        return any(before.get(m) != after.get(m) for m in before.keys())
