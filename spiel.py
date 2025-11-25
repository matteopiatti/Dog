# train_dqn_param_batched.py

import os
import numpy as np
import pyspiel

from DogGame import DogGame, NUM_ACTIONS
from open_spiel.python import rl_environment
from open_spiel.python.algorithms import random_agent

from dqn_parametric import ParametricDQNAgent
from action_features import encode_action_features, ACT_DIM

# --------------------------------------------------
# hyperparameters
# --------------------------------------------------

NUM_EPISODES = 50_000
MODEL_PATH = "dog_dqn_param.pt"

# train only every N transitions
TRAIN_EVERY = 32  # how many transitions between training calls
TRAIN_ITERS = 4  # how many gradient steps when we train
LOG_EVERY_EP = 100

# --------------------------------------------------
# env / game setup
# --------------------------------------------------

game = DogGame()
env = rl_environment.Environment(game=game)
num_players = game.num_players()

# infer observation size
dummy_ts = env.reset()
obs_dim = len(dummy_ts.observations["info_state"][0])

# --------------------------------------------------
# agent
# --------------------------------------------------

agent = ParametricDQNAgent(
    player_id=0,
    obs_dim=obs_dim,
    act_dim=ACT_DIM,
)

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

# stats
wins = 0
losses = 0
draws = 0

global_step = 0  # count player-0 transitions

# --------------------------------------------------
# helper
# --------------------------------------------------


def team_progress(inner) -> tuple[float, float, float]:
    board = inner.board
    p0 = inner.players[0]
    teammate = inner.teammate(p0)
    team = (p0, teammate)

    finished = float(board.team_finished_marbles(team))

    home_count = 0
    for p in team:
        home_count += sum(1 for m in board.home[p] if m is not None)
    home_count = float(home_count)

    distance = float(
        board.total_distance_to_home(team[0]) + board.total_distance_to_home(team[1])
    )

    return finished, home_count, distance


# --------------------------------------------------
# training loop
# --------------------------------------------------

for ep in range(NUM_EPISODES):
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
            # --- state & progress BEFORE move ---
            obs = np.array(time_step.observations["info_state"][0], dtype=np.float32)
            state_before = env.get_state
            inner_before = state_before._inner
            fin_b, home_b, dist_b = team_progress(inner_before)

            # --- build legal actions & features ---
            legal_ids = time_step.observations["legal_actions"][0]
            state_for_actions = state_before  # same DogState
            legal_act_feats = [
                encode_action_features(state_for_actions, 0, aid) for aid in legal_ids
            ]

            # choose action
            idx, chosen_feat = agent.select_action(
                obs, legal_act_feats, eval_mode=False
            )
            chosen_id = legal_ids[idx]

            # --- apply action ---
            next_time_step = env.step([chosen_id])

            # --- state & progress AFTER move ---
            state_after = env.get_state
            inner_after = state_after._inner
            fin_a, home_a, dist_a = team_progress(inner_after)

            # progress deltas
            df = fin_a - fin_b  # finished marbles gained
            dh = home_a - home_b  # extra marbles moved into home rows
            dd = dist_b - dist_a  # positive if closer to home

            # shaping reward
            NUM_FIELDS = inner_after.board.NUM_FIELDS
            norm_dist = NUM_FIELDS * 8.0
            progress_reward = (
                1.0 * df  # +1 per finished marble
                + 0.2 * dh  # +0.2 per new marble in home row
                + 0.5 * (dd / norm_dist)  # small reward for moving closer
            )

            env_reward = next_time_step.rewards[0]

            r = progress_reward + env_reward
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
        reward_list[-1] += outcome
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

    # push transitions into replay and train batched
    for obs, af, r, nxt_obs, nxt_af, d in zip(
        obs_list,
        act_feat_list,
        reward_list,
        next_obs_list,
        next_act_feat_list,
        done_list,
    ):
        agent.store_transition(obs, af, r, nxt_obs, nxt_af, d)
        global_step += 1

        if global_step % TRAIN_EVERY == 0:
            for _ in range(TRAIN_ITERS):
                agent.train_step()

    # periodic logging and checkpoint
    if (ep + 1) % LOG_EVERY_EP == 0:
        total = wins + losses + draws
        win_rate = wins / total if total > 0 else 0.0
        print(
            f"Episode {ep+1}: wins={wins}, losses={losses}, draws={draws}, "
            f"win_rate={win_rate:.3f}"
        )
        wins = 0
        losses = 0
        draws = 0
        agent.save(MODEL_PATH)

# final save
agent.save(MODEL_PATH)
print("Training finished, model saved.")
