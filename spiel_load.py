# play_vs_dqn.py
import numpy as np
import torch
from DogGame import DogGame, NUM_ACTIONS
from open_spiel.python import rl_environment
from open_spiel.python.algorithms import random_agent
from dqn_agent import DQNAgent
from dog.cli import render

game = DogGame()
env = rl_environment.Environment(game=game)
num_players = game.num_players()

dummy_ts = env.reset()
obs_dim = len(dummy_ts.observations["info_state"][0])

dqn = DQNAgent(
    player_id=0,
    obs_dim=obs_dim,
    num_actions=NUM_ACTIONS,
)
dqn.load("dog_dqn.pt", map_location=torch.device("cpu"))

# others random just as example
opponents = [
    random_agent.RandomAgent(pid, NUM_ACTIONS, name=f"random_{pid}")
    for pid in range(1, num_players)
]

time_step = env.reset()
while not time_step.last():
    # optional: render underlying DogGame state
    dog_state = env.get_state()  # pyspiel.State
    inner = dog_state._inner  # your DogGameState
    render(inner)

    current_player = time_step.observations["current_player"]

    if current_player == 0:
        obs = np.array(time_step.observations["info_state"][0], dtype=np.float32)
        legal_actions = time_step.observations["legal_actions"][0]
        action = dqn.select_action(obs, legal_actions, eval_mode=True)
        agent_action = action
    else:
        out = opponents[current_player - 1].step(time_step, is_evaluation=True)
        agent_action = out.action

    time_step = env.step([agent_action])

print("Game finished. Returns:", time_step.rewards)
