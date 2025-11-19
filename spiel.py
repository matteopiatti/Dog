# train_dqn_param.py
import numpy as np
import pyspiel
import os

from DogGame import DogGame, NUM_ACTIONS
from open_spiel.python import rl_environment
from open_spiel.python.algorithms import random_agent

from dqn_parametric import ParametricDQNAgent
from action_features import encode_action_features, ACT_DIM

game = DogGame()
env = rl_environment.Environment(game=game)
num_players = game.num_players()

# infer observation size
dummy_ts = env.reset()
obs_dim = len(dummy_ts.observations["info_state"][0])

agent = ParametricDQNAgent(
    player_id=0,
    obs_dim=obs_dim,
    act_dim=ACT_DIM,
)

MODEL_PATH = "dog_dqn_param.pt"
if os.path.exists(MODEL_PATH):
    agent.load(MODEL_PATH)
    print("Loaded existing parametric DQN checkpoint.")
else:
    print("No existing checkpoint, starting fresh.")

# opponents as random agents
opponents = [
    random_agent.RandomAgent(pid, NUM_ACTIONS, name=f"random_{pid}")
    for pid in range(1, num_players)
]

num_episodes = 10_000
wins = 0
losses = 0
draws = 0

for ep in range(num_episodes):
    time_step = env.reset()

    # episode storage for player-0 moves
    obs_list = []
    act_feat_list = []
    reward_list = []
    next_obs_list = []
    done_list = []

    while not time_step.last():
        current_player = time_step.observations["current_player"]

        if current_player == 0:
            # current state + DogState
            obs = np.array(time_step.observations["info_state"][0], dtype=np.float32)
            state = env.get_state
            legal_ids = time_step.observations["legal_actions"][0]

            # build features for each legal action id
            legal_act_feats = [
                encode_action_features(state, 0, aid) for aid in legal_ids
            ]

            # choose action (index in legal list)
            idx, chosen_feat = agent.select_action(
                obs, legal_act_feats, eval_mode=False
            )
            chosen_id = legal_ids[idx]

            next_time_step = env.step([chosen_id])

            r = next_time_step.rewards[0]
            d = next_time_step.last()
            next_obs = np.array(
                next_time_step.observations["info_state"][0], dtype=np.float32
            )

            obs_list.append(obs)
            act_feat_list.append(chosen_feat)
            reward_list.append(r)
            next_obs_list.append(next_obs)
            done_list.append(d)

            time_step = next_time_step

        else:
            # opponents act
            out = opponents[current_player - 1].step(time_step, is_evaluation=False)
            time_step = env.step([out.action])

    # terminal bookkeeping for opponents
    for opp in opponents:
        opp.step(time_step, is_evaluation=False)

    # decide outcome from winner team
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

    # overwrite last reward with outcome
    if reward_list:
        reward_list[-1] = outcome
        done_list[-1] = True  # last transition is terminal

    # build next_act_feat_list for SARSA: a_{t+1} = act_feat_{t+1}
    # for last step, next_act_feat is zero vector
    zero_act = np.zeros(ACT_DIM, dtype=np.float32)
    next_act_feat_list = []
    for i in range(len(act_feat_list)):
        if i + 1 < len(act_feat_list):
            next_act_feat_list.append(act_feat_list[i + 1])
        else:
            next_act_feat_list.append(zero_act)

    # push transitions into replay and train
    for obs, af, r, nxt_obs, nxt_af, d in zip(
        obs_list,
        act_feat_list,
        reward_list,
        next_obs_list,
        next_act_feat_list,
        done_list,
    ):
        agent.store_transition(obs, af, r, nxt_obs, nxt_af, d)
        agent.train_step()

    if (ep + 1) % 100 == 0:
        print(
            f"Episode {ep+1}: wins={wins}, losses={losses}, draws={draws}, win_rate={wins/(wins+losses+draws):.3f}"
        )
        agent.save(MODEL_PATH)
