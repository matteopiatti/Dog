# train_dqn.py
import numpy as np
import pyspiel
import os

from DogGame import DogGame, NUM_ACTIONS
from open_spiel.python import rl_environment
from open_spiel.python.algorithms import random_agent

from dqn_agent import DQNAgent

game = DogGame()
env = rl_environment.Environment(game=game)
num_players = game.num_players()
num_actions = NUM_ACTIONS

# infer observation size from env
dummy_ts = env.reset()
obs_example = dummy_ts.observations["info_state"][0]
obs_dim = len(obs_example)

dqn = DQNAgent(
    player_id=0,
    obs_dim=obs_dim,
    num_actions=num_actions,
    lr=5e-4,
    gamma=0.99,
    epsilon_start=1.0,
    epsilon_end=0.05,
    epsilon_decay_steps=100_000,
    buffer_capacity=200_000,
    batch_size=128,
    target_update_freq=1000,
)

if os.path.exists("dog_dqn.pt"):
    dqn.load("dog_dqn.pt")
    print("Loaded existing DQN checkpoint.")
else:
    print("No existing checkpoint, starting fresh.")

# opponents as random agents
opponents = []
for pid in range(1, num_players):
    opponents.append(
        random_agent.RandomAgent(
            player_id=pid, num_actions=num_actions, name=f"random_{pid}"
        )
    )

num_episodes = 10_000
wins = 0
losses = 0
draws = 0


for ep in range(num_episodes):
    time_step = env.reset()
    episode_steps = []  # will store (obs, action, next_obs, reward, done)

    while not time_step.last():
        current_player = time_step.observations["current_player"]

        if current_player == 0:
            obs = np.array(time_step.observations["info_state"][0], dtype=np.float32)
            legal_actions = time_step.observations["legal_actions"][0]
            action = dqn.select_action(obs, legal_actions, eval_mode=False)

            next_time_step = env.step([action])

            # temporary reward/done, will fix last one after episode
            reward = next_time_step.rewards[0]  # usually 0
            done = next_time_step.last()
            next_obs = np.array(
                next_time_step.observations["info_state"][0], dtype=np.float32
            )

            episode_steps.append((obs, action, next_obs, reward, done))
            time_step = next_time_step

        else:
            out = opponents[current_player - 1].step(time_step, is_evaluation=False)
            time_step = env.step([out.action])

    # opponents terminal bookkeeping
    for opp in opponents:
        opp.step(time_step, is_evaluation=False)

    # determine final outcome from winner, not rewards
    state = env.get_state
    inner = state._inner
    winner_team = inner.winner

    if winner_team is None:
        outcome = 0.0
        draws += 1
    elif inner.players[0] in winner_team:
        outcome = 1.0
        wins += 1
    else:
        outcome = -1.0
        losses += 1

    # if player 0 moved at least once this episode, fix last transition
    if episode_steps:
        obs_l, act_l, next_obs_l, _, _ = episode_steps[-1]
        episode_steps[-1] = (obs_l, act_l, next_obs_l, outcome, True)

    # now store all transitions and train
    for obs, action, next_obs, reward, done in episode_steps:
        dqn.store_transition(obs, action, next_obs, reward, done)
        dqn.train_step()

    if (ep + 1) % 100 == 0:
        print(
            f"Episode {ep+1}: wins={wins}, losses={losses}, draws={draws}, winrate={wins/(wins+losses+draws):.3f}"
        )
        dqn.save("dog_dqn.pt")
