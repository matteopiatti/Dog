import pyspiel
import numpy as np
from typing import List
from dog.engine_rl import (
    GameState as DogGameState,
    setup_game_rl,
    auto_advance_to_decision,
)
from dog.move import start_action
from dog.enums import GamePhase
from dog.rules import legal_actions

MAX_HAND_SIZE = 6
MAX_ACTIONS_PER_TURN = 64
NUM_SWITCH_ACTIONS = MAX_HAND_SIZE
NUM_PLAY_ACTIONS = MAX_ACTIONS_PER_TURN
NUM_ACTIONS = NUM_SWITCH_ACTIONS + NUM_PLAY_ACTIONS


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
            provides_information_state_tensor=False,
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

        # DEAL and TURN should not be decision points if auto_advance works
        return []

    def apply_action(self, action: int) -> None:
        # Decode action depending on phase
        phase = self._inner.phase

        if phase == GamePhase.SWITCH:
            idx = action  # 0..hand_size-1
            player = self._inner.current_player
            card = player.hand[idx]
            self._inner.switch_cards.append((player, card))

            # move to next player or execute switch like engine.step
            if len(self._inner.switch_cards) < len(self._inner.players):
                self._inner.advance_player()
            else:
                # execute switch
                for player_from, card in self._inner.switch_cards:
                    self._inner.teammate(player_from).receive_card(card)
                    player_from.play_card(card)
                self._inner.switch_cards.clear()
                self._inner.phase = GamePhase.TURN

        elif phase == GamePhase.PLAY:
            idx = action - NUM_SWITCH_ACTIONS
            acts = getattr(self, "_cached_actions", None)
            if acts is None or idx >= len(acts):
                raise ValueError("Invalid action index")
            a = acts[idx]
            # PLAY branch of engine.step, but without agent
            start_action(self._inner, a)
            self._inner.current_player.play_card(a.card)
            self._inner.reset_actions()
            self._inner.advance_player()
            self._inner.phase = GamePhase.TURN

        else:
            raise RuntimeError(f"apply_action called in non-decision phase {phase}")

        # After applying an action, auto-advance to next decision or terminal
        auto_advance_to_decision(self._inner)

    # ----- Observation -----

    def observation_tensor(self, player=None):
        # Very simple feature vector: you can improve later.
        # Example: track positions of marbles and hand sizes.
        if player is None:
            player = self.current_player()
        num_players = len(self._inner.players)

        # tensor size: track occupancy + home + hand_sizes
        track = self._inner.board.track
        board_vec = np.zeros(
            self.game.get_game_info().num_players * len(track), dtype=np.float32
        )
        for i, (m, p) in enumerate(track):
            if p is None:
                continue
            pid = self._inner.players.index(p)
            board_vec[pid * len(track) + i] = 1.0

        hand_sizes = np.array(
            [len(p.hand) for p in self._inner.players], dtype=np.float32
        )

        obs = np.concatenate([board_vec, hand_sizes], axis=0)
        return obs

    def observation_string(self, player=None):
        # simple text representation
        return f"Phase={self._inner.phase.name}, CP={self._inner.current_player.name}"

    # ----- Cloning -----

    def clone(self):
        return DogState(self.get_game(), self._inner.clone())
