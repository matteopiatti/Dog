import pyspiel
import numpy as np
from typing import List
from dog.engine_rl import (
    GameState as DogGameState,
    setup_game_rl,
    auto_advance_to_decision,
)
from dog.move import move_marble, start_action
from dog.enums import GamePhase, MoveKind
from dog.rules import legal_actions, marble_allowed_steps
from dog.cards import Card, CardType

MAX_HAND_SIZE = 6
MAX_ACTIONS_PER_TURN = 500
MAX_SPLIT_OPTIONS = 32

NUM_SWITCH_ACTIONS = MAX_HAND_SIZE
NUM_PLAY_ACTIONS = MAX_ACTIONS_PER_TURN
NUM_SPLIT_ACTIONS = MAX_SPLIT_OPTIONS

OFFSET_SWITCH = 0
OFFSET_PLAY = OFFSET_SWITCH + NUM_SWITCH_ACTIONS
OFFSET_SPLIT = OFFSET_PLAY + NUM_PLAY_ACTIONS

NUM_ACTIONS = NUM_SWITCH_ACTIONS + NUM_PLAY_ACTIONS + NUM_SPLIT_ACTIONS


class DogGame(pyspiel.Game):
    def __init__(self, params=None):
        game_type = pyspiel.GameType(
            short_name="dog",
            long_name="Dog",
            dynamics=pyspiel.GameType.Dynamics.SEQUENTIAL,
            chance_mode=pyspiel.GameType.ChanceMode.DETERMINISTIC,
            information=pyspiel.GameType.Information.PERFECT_INFORMATION,
            utility=pyspiel.GameType.Utility.GENERAL_SUM,
            reward_model=pyspiel.GameType.RewardModel.TERMINAL,
            max_num_players=4,
            min_num_players=4,
            provides_information_state_string=False,
            provides_information_state_tensor=True,
            provides_observation_string=True,
            provides_observation_tensor=True,
            parameter_specification={},
        )
        game_info = pyspiel.GameInfo(
            num_distinct_actions=NUM_ACTIONS,
            max_chance_outcomes=0,
            num_players=4,
            min_utility=-1.0,  # terminal reward range
            max_utility=1.0,
            utility_sum=None,  # None for general-sum games
            max_game_length=500,
        )
        super().__init__(game_type, game_info, params or {})

    def new_initial_state(self):
        inner = setup_game_rl(num_players=4)
        auto_advance_to_decision(inner)
        return DogState(self, inner)

    def make_copy(self, state):
        return DogState(self, state._inner.clone())


class DogState(pyspiel.State):
    def __init__(self, game: DogGame, inner: DogGameState):
        super().__init__(game)
        self._inner = inner

    # ----- Core API -----

    def current_player(self) -> int:
        if self._inner.finished:
            return pyspiel.PlayerId.TERMINAL
        return self._inner.players.index(self._inner.current_player)

    def is_terminal(self) -> bool:
        return self._inner.finished

    def rewards(self) -> List[float]:
        # per-move rewards; we use only terminal returns here
        if not self._inner.finished:
            return [0.0] * 4
        r = [0.0] * 4
        if self._inner.winner is None:
            return r
        winners = self._inner.winner  # tuple(Player, Player)
        for p in self._inner.players:
            idx = self._inner.players.index(p)
            if p in winners:
                r[idx] = 1.0
            else:
                r[idx] = -1.0
        return r

    def returns(self) -> List[float]:
        return self.rewards()

    def legal_actions(self, player=None) -> List[int]:
        if self.is_terminal():
            return []
        if player is None:
            player = self.current_player()
        if player != self.current_player():
            return []

        phase = self._inner.phase

        if phase == GamePhase.SWITCH:
            hand = self._inner.current_player.hand
            return [i for i in range(len(hand))]  # 0..len(hand)-1

        if phase == GamePhase.PLAY:
            acts = legal_actions(self._inner)
            # cache actions for decoding
            self._cached_actions = acts
            base = NUM_SWITCH_ACTIONS
            return [base + i for i in range(len(acts))]

        if phase == GamePhase.SPLIT:
            steps_left = self._inner.split_steps_remaining
            allowed = marble_allowed_steps(self._inner, steps_left)
            # flatten (marble, step) options into a list
            options = []
            for marble, steps in allowed.items():
                for s in steps:
                    options.append((marble, s))
            self._split_options = options  # cache for decoding

            # encode them as OFFSET_SPLIT + index
            return [OFFSET_SPLIT + i for i in range(len(options))]
        return []

    def apply_action(self, action: int) -> None:
        phase = self._inner.phase

        if phase == GamePhase.SWITCH:
            idx = action - OFFSET_SWITCH
            player = self._inner.current_player
            card = player.hand[idx]
            self._inner.switch_cards.append((player, card))
            if len(self._inner.switch_cards) < len(self._inner.players):
                self._inner.advance_player()
            else:
                for player_from, card in self._inner.switch_cards:
                    self._inner.teammate(player_from).receive_card(card)
                    player_from.play_card(card)
                self._inner.switch_cards.clear()
                self._inner.phase = GamePhase.TURN

        elif phase == GamePhase.PLAY:
            idx = action - OFFSET_PLAY
            acts = getattr(self, "_cached_actions", None)
            if acts is None or idx >= len(acts):
                raise ValueError("Invalid PLAY action index")
            a = acts[idx]

            if a.kind == MoveKind.SPLIT:
                # start the split; do NOT consume card or advance player yet
                start_action(self._inner, a)  # sets phase=SPLIT, steps_remaining=7
                # keep same current_player
            else:
                # normal move
                start_action(self._inner, a)
                self._inner.current_player.play_card(a.card)
                self._inner.reset_actions()
                self._inner.advance_player()
                self._inner.phase = GamePhase.TURN

        elif phase == GamePhase.SPLIT:
            idx = action - OFFSET_SPLIT
            opts = getattr(self, "_split_options", None)
            if opts is None or idx >= len(opts):
                raise ValueError("Invalid SPLIT sub-action index")
            marble, step = opts[idx]

            # perform this sub-move
            move_marble(self._inner, marble, step)
            self._inner.split_steps_remaining -= step

            # check if we can continue splitting
            steps_left = self._inner.split_steps_remaining
            if steps_left > 0:
                allowed = marble_allowed_steps(self._inner, steps_left)
                if allowed:
                    # still SPLIT phase, same player
                    self._inner.phase = GamePhase.SPLIT
                else:
                    # no more possible split moves
                    self._finish_split_turn()
            else:
                # exactly used up 7
                self._finish_split_turn()

        else:
            raise RuntimeError(f"apply_action called in non-decision phase {phase}")

        # move to next decision / terminal
        auto_advance_to_decision(self._inner)

    # ----- Observation -----

    def observation_tensor(self, player=None):
        if player is None:
            player = self.current_player()

        inner = self._inner
        board = inner.board
        players = inner.players
        num_players = len(players)

        # ---------- 1) Board occupancy (same as before) ----------
        track = board.track
        board_vec = np.zeros(num_players * len(track), dtype=np.float32)
        for i, (m, p) in enumerate(track):
            if p is None:
                continue
            pid = players.index(p)
            board_vec[pid * len(track) + i] = 1.0

        # ---------- 2) Phase one-hot ----------
        phase_vec = np.zeros(len(GamePhase), dtype=np.float32)
        phase_index = list(GamePhase).index(inner.phase)
        phase_vec[phase_index] = 1.0

        # ---------- 3) Current player one-hot ----------
        cp_vec = np.zeros(num_players, dtype=np.float32)
        cp_idx = players.index(inner.current_player)
        cp_vec[cp_idx] = 1.0

        # ---------- 4) Marble stats per player ----------
        # For each player: [num_in_play, num_finished, total_dist_to_home]
        # Normalize counts by 4, distance by (NUM_FIELDS * 4)
        marble_stats = []
        max_dist = board.NUM_FIELDS * 4.0
        for p in players:
            in_play = len(board.player_marbles_in_play(p)) / 4.0
            finished = board.player_finished_marbles(p) / 4.0
            total_dist = board.total_distance_to_home(p) / max_dist
            marble_stats.extend([in_play, finished, total_dist])
        marble_stats = np.array(marble_stats, dtype=np.float32)

        # ---------- 5) Hand sizes per player (normalized) ----------
        hand_sizes = (
            np.array([len(pl.hand) for pl in players], dtype=np.float32) / 6.0
        )  # max hand size = 6

        # ---------- 6) Current player's card rank counts ----------
        # vector over CardType (including JOKER), counts normalized by 6
        rank_list = list(CardType)
        rank_vec = np.zeros(len(rank_list), dtype=np.float32)
        current_pl = players[player]
        for c in current_pl.hand:
            r_idx = rank_list.index(c.rank)
            rank_vec[r_idx] += 1.0
        rank_vec /= 6.0  # normalize

        # ---------- 7) Concatenate everything ----------
        obs = np.concatenate(
            [
                board_vec,
                phase_vec,
                cp_vec,
                marble_stats,
                hand_sizes,
                rank_vec,
            ],
            axis=0,
        )
        return obs

    def observation_string(self, player=None):
        # simple text representation
        return f"Phase={self._inner.phase.name}, CP={self._inner.current_player.name}"

    def _finish_split_turn(self):
        # consume the original 7 card now, finish turn
        player = self._inner.current_player
        if self._inner.split_card is not None:
            player.play_card(self._inner.split_card)
        self._inner.split_card = None
        self._inner.split_steps_remaining = 0
        self._inner.reset_actions()
        self._inner.advance_player()
        self._inner.phase = GamePhase.TURN

    def information_state_tensor(self, player):
        # simplest: reuse observation encoding for that player
        return self.observation_tensor(player)

    def clone(self):
        return DogState(self.get_game(), self._inner.clone())
