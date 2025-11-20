# train_dqn_param_parallel.py

import os
import numpy as np
import pyspiel

from DogGame import DogGame, NUM_ACTIONS
from open_spiel.python import rl_environment
from open_spiel.python.algorithms import random_agent

from dqn_parametric import ParametricDQNAgent
from action_features import encode_action_features, ACT_DIM

# -------------------------
# hyperparameters
# -------------------------

NUM_ENVS = 8  # number of parallel envs in one process
NUM_EPISODES = 50_000
TRAIN_EVERY = 32  # train every N transitions
TRAIN_ITERS = 4  # number of train_step() calls when training
MODEL_PATH = "dog_dqn_param.pt"
LOG_EVERY_EP = 100

# -------------------------
# helper functions
# -------------------------


def team_progress(inner) -> tuple[float, float, float]:
    board = inner.board
    p0 = inner.players[0]
    teammate = inner.teammate(p0)
    team = (p0, teammate)

    finished = float(board.team_finished_marbles(team))

    home_count = 0.0
    for p in team:
        home_count += sum(1 for m in board.home[p] if m is not None)
    home_count = float(home_count)

    distance = float(
        board.total_distance_to_home(team[0]) + board.total_distance_to_home(team[1])
    )

    return finished, home_count, distance


# -------------------------
# setup single dummy env to infer sizes
# -------------------------

dummy_game = DogGame()
dummy_env = rl_environment.Environment(game=dummy_game)
dummy_ts = dummy_env.reset()
obs_dim = len(dummy_ts.observations["info_state"][0])

num_players = dummy_game.num_players()

# -------------------------
# agent
# -------------------------

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

# -------------------------
# vectorized environments
# -------------------------

games = [DogGame() for _ in range(NUM_ENVS)]
envs = [rl_environment.Environment(game=g) for g in games]

# opponents per environment
opponents_list = [
    [
        random_agent.RandomAgent(pid, NUM_ACTIONS, name=f"random_{pid}_env{env_idx}")
        for pid in range(1, g.num_players())
    ]
    for env_idx, g in enumerate(games)
]

time_steps = [env.reset() for env in envs]

# per-env episode storage
obs_lists = [[] for _ in range(NUM_ENVS)]
act_feat_lists = [[] for _ in range(NUM_ENVS)]
reward_lists = [[] for _ in range(NUM_ENVS)]
next_obs_lists = [[] for _ in range(NUM_ENVS)]
done_lists = [[] for _ in range(NUM_ENVS)]

# stats
global_step = 0
completed_episodes = 0
wins = 0
losses = 0
draws = 0

# -------------------------
# main loop
# -------------------------

while completed_episodes < NUM_EPISODES:
    # step each environment once
    for env_idx, (env, ts) in enumerate(zip(envs, time_steps)):
        if completed_episodes >= NUM_EPISODES:
            break

        # if episode finished, finalize and reset
        if ts.last():
            # terminal bookkeeping for opponents
            opponents = opponents_list[env_idx]
            for opp in opponents:
                opp.step(ts, is_evaluation=False)

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
            if reward_lists[env_idx]:
                reward_lists[env_idx][-1] += outcome
                done_lists[env_idx][-1] = True

            # build next_act_feat_list (SARSA style)
            zero_act = np.zeros(ACT_DIM, dtype=np.float32)
            next_act_feat_list = []
            af_list = act_feat_lists[env_idx]
            for i in range(len(af_list)):
                if i + 1 < len(af_list):
                    next_act_feat_list.append(af_list[i + 1])
                else:
                    next_act_feat_list.append(zero_act)

            # push transitions and train periodically
            for obs, af, r, nxt_obs, nxt_af, d in zip(
                obs_lists[env_idx],
                act_feat_lists[env_idx],
                reward_lists[env_idx],
                next_obs_lists[env_idx],
                next_act_feat_list,
                done_lists[env_idx],
            ):
                agent.store_transition(obs, af, r, nxt_obs, nxt_af, d)
                global_step += 1

                if global_step % TRAIN_EVERY == 0:
                    for _ in range(TRAIN_ITERS):
                        agent.train_step()

            # reset episode buffers
            obs_lists[env_idx].clear()
            act_feat_lists[env_idx].clear()
            reward_lists[env_idx].clear()
            next_obs_lists[env_idx].clear()
            done_lists[env_idx].clear()

            # increment episode count and maybe log
            completed_episodes += 1

            if completed_episodes > 0 and completed_episodes % LOG_EVERY_EP == 0:
                total = wins + losses + draws
                win_rate = wins / total if total > 0 else 0.0
                print(
                    f"Episode {completed_episodes}: "
                    f"wins={wins}, losses={losses}, draws={draws}, win_rate={win_rate:.3f}"
                )
                wins = 0
                losses = 0
                draws = 0
                agent.save(MODEL_PATH)

            # if we still need more episodes, reset env
            if completed_episodes < NUM_EPISODES:
                ts = env.reset()
                time_steps[env_idx] = ts
            continue

        # non-terminal: advance one step
        current_player = ts.observations["current_player"]

        if current_player == 0:
            # state & progress before move
            obs = np.array(ts.observations["info_state"][0], dtype=np.float32)
            state_before = env.get_state
            inner_before = state_before._inner
            fin_b, home_b, dist_b = team_progress(inner_before)

            # legal actions and features
            legal_ids = ts.observations["legal_actions"][0]
            state_for_actions = state_before
            legal_act_feats = [
                encode_action_features(state_for_actions, 0, aid) for aid in legal_ids
            ]

            # choose action
            idx, chosen_feat = agent.select_action(
                obs, legal_act_feats, eval_mode=False
            )
            chosen_id = legal_ids[idx]

            # apply action
            next_ts = env.step([chosen_id])

            # state & progress after move
            state_after = env.get_state
            inner_after = state_after._inner
            fin_a, home_a, dist_a = team_progress(inner_after)

            df = fin_a - fin_b
            dh = home_a - home_b
            dd = dist_b - dist_a

            NUM_FIELDS = inner_after.board.NUM_FIELDS
            norm_dist = NUM_FIELDS * 8.0

            progress_reward = 1.0 * df + 0.2 * dh + 0.5 * (dd / norm_dist)

            env_reward = next_ts.rewards[0]

            r = progress_reward + env_reward
            d = next_ts.last()
            next_obs = np.array(next_ts.observations["info_state"][0], dtype=np.float32)

            obs_lists[env_idx].append(obs)
            act_feat_lists[env_idx].append(chosen_feat)
            reward_lists[env_idx].append(r)
            next_obs_lists[env_idx].append(next_obs)
            done_lists[env_idx].append(d)

            time_steps[env_idx] = next_ts

        else:
            opponents = opponents_list[env_idx]
            out = opponents[current_player - 1].step(ts, is_evaluation=False)
            next_ts = env.step([out.action])
            time_steps[env_idx] = next_ts

# final save
agent.save(MODEL_PATH)
print("Training finished, model saved.")
