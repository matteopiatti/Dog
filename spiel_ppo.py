# train_ppo_param.py

import os
import numpy as np
import pyspiel

from DogGame import DogGame, NUM_ACTIONS
from open_spiel.python import rl_environment
from open_spiel.python.algorithms import random_agent

from parametric_ppo import ParametricPPOAgent
from action_features import encode_action_features, ACT_DIM

# --------------------------------------------------
# hyperparameters
# --------------------------------------------------

NUM_EPISODES = 50_000
MODEL_PATH = "dog_param_ppo.pt"

UPDATE_EVERY_EP = 10
LOG_EVERY_EP = 100

# --------------------------------------------------
# env / game setup
# --------------------------------------------------

game = DogGame()
env = rl_environment.Environment(game=game)
num_players = game.num_players()

dummy_ts = env.reset()
obs_dim = len(dummy_ts.observations["info_state"][0])

# --------------------------------------------------
# agent
# --------------------------------------------------

agent = ParametricPPOAgent(
    obs_dim=obs_dim,
    act_dim=ACT_DIM,
)

if os.path.exists(MODEL_PATH):
    agent.load(MODEL_PATH)
    print("Loaded existing PPO checkpoint.")
else:
    print("No existing checkpoint, starting fresh.")

# opponents as random agents
opponents = [
    random_agent.RandomAgent(pid, NUM_ACTIONS, name=f"random_{pid}")
    for pid in range(1, num_players)
]

wins = 0
losses = 0
draws = 0

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
    agent.current_episode = []  # ensure clean

    while not time_step.last():
        current_player = time_step.observations["current_player"]

        if current_player == 0:
            obs = np.array(time_step.observations["info_state"][0], dtype=np.float32)

            state_before = env.get_state
            inner_before = state_before._inner
            fin_b, home_b, dist_b = team_progress(inner_before)

            # build legal actions and features
            legal_ids = time_step.observations["legal_actions"][0]
            state_for_actions = state_before
            legal_act_feats = [
                encode_action_features(state_for_actions, 0, aid) for aid in legal_ids
            ]

            # select action via PPO policy
            idx, chosen_feat, logp, value = agent.select_action(
                obs, legal_act_feats, eval_mode=False
            )
            chosen_id = legal_ids[idx]

            next_time_step = env.step([chosen_id])

            state_after = env.get_state
            inner_after = state_after._inner
            fin_a, home_a, dist_a = team_progress(inner_after)

            df = fin_a - fin_b
            dh = home_a - home_b
            dd = dist_b - dist_a

            NUM_FIELDS = inner_after.board.NUM_FIELDS
            norm_dist = NUM_FIELDS * 8.0
            progress_reward = 1.0 * df + 0.2 * dh + 0.5 * (dd / norm_dist)

            env_reward = next_time_step.rewards[0]
            r = progress_reward + env_reward
            d = next_time_step.last()

            agent.store_step(
                obs=obs,
                legal_act_feats=legal_act_feats,
                action_idx=idx,
                logp=logp,
                value=value,
                reward=r,
                done=d,
            )

            time_step = next_time_step

        else:
            out = opponents[current_player - 1].step(time_step, is_evaluation=False)
            time_step = env.step([out.action])

    # terminal bookkeeping for opponents
    for opp in opponents:
        opp.step(time_step, is_evaluation=False)

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

    # add outcome to final step reward, if any
    if agent.current_episode:
        agent.current_episode[-1].reward += outcome
        agent.current_episode[-1].done = True

    # move episode into global buffer with GAE/returns
    agent.finish_episode()

    # PPO update every few episodes
    if (ep + 1) % UPDATE_EVERY_EP == 0:
        agent.update()

    # logging and checkpoint
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

agent.save(MODEL_PATH)
print("Training finished, model saved.")
