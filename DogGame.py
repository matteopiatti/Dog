import pyspiel

_NUM_PLAYERS = 4
_NUM_DISTINCT_ACTIONS = 500
_MAX_CHANCE_OUTCOMES = 52
_MAX_GAME_LENGTH = 1000

_GAME_TYPE = pyspiel.GameType(
    short_name="dog_game",
    long_name="Dog Game",
    dynamics=pyspiel.GameType.Dynamics.SEQUENTIAL,
    chance_mode=pyspiel.GameType.ChanceMode.EXTERNAL_STOCHASTIC,
    information=pyspiel.GameType.Information.IMPERFECT_INFORMATION,
    utility=pyspiel.GameType.Utility.ZERO_SUM,
    reward_model=pyspiel.GameType.RewardModel.TERMINAL,
    max_num_players=_NUM_PLAYERS,
    min_num_players=_NUM_PLAYERS,
    provides_information_state_string=True,
    provides_information_state_tensor=False,
    provides_observation_string=True,
    provides_observation_tensor=False,
    parameter_specification={},
)
_GAME_INFO = pyspiel.GameInfo(
    num_distinct_actions=_NUM_DISTINCT_ACTIONS,
    max_chance_outcomes=_MAX_CHANCE_OUTCOMES,
    num_players=_NUM_PLAYERS,
    min_utility=0.0,
    max_utility=1.0,
    utility_sum=0.0,
    max_game_length=_MAX_GAME_LENGTH,
)


class DogGame(pyspiel.Game):
    """A Python version of the Dog card game."""

    def __init__(self, params=None):
        super().__init__(_GAME_TYPE, _GAME_INFO, params or dict())

    def new_initial_state(self):
        """Returns a state corresponding to the start of a game."""
        return DogGameState(self)


class DogGameState(pyspiel.State):
    """A python version of the Dog game state."""

    def __init__(self, game):
        super().__init__(game)
        self._cur_player = 0
