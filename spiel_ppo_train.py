# train_ppo_param.py
# self-play training: current agent team vs frozen old snapshot team
# snapshot is updated only if current agent beats it >= 80% (no-draw win rate)

import os
import numpy as np
import pyspiel
import torch

from DogGame import DogGame, NUM_ACTIONS
from open_spiel.python import rl_environment


from parametric_ppo import ParametricPPOAgent
from action_features import encode_action_features, ACT_DIM

# --------------------------------------------------
# hyperparameters
# --------------------------------------------------

NUM_EPISODES = 50_000
MODEL_PATH = "dog_param_ppo.pt"

UPDATE_EVERY_EP = 10
LOG_EVERY_EP = 100

EVAL_EVERY_EP = 100  # how often to evaluate vs snapshot
EVAL_EPISODES = 50  # episodes per eval
SNAPSHOT_THRESHOLD = 0.80  # win rate (no draws) to replace snapshot

# --------------------------------------------------
# env / game setup
# --------------------------------------------------

game = DogGame()
env = rl_environment.Environment(game=game)
num_players = game.num_players()

dummy_ts = env.reset()
obs_dim = len(dummy_ts.observations["info_state"][0])

# determine teammate id for player 0 from underlying game state
state0 = env.get_state
inner0 = state0._inner
p0 = inner0.players[0]
teammate_player = inner0.teammate(p0)
teammate_id = inner0.players.index(teammate_player)
print(f"Player 0 teammate id: {teammate_id}")

# --------------------------------------------------
# agents
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

# frozen snapshot opponent
old_agent = ParametricPPOAgent(
    obs_dim=obs_dim,
    act_dim=ACT_DIM,
)
# initialize snapshot = current
old_agent.policy.load_state_dict(agent.policy.state_dict())
old_agent.value_net.load_state_dict(agent.value_net.state_dict())
print("Initialized snapshot opponent from current agent.")

with torch.no_grad():
    max_diff = 0.0
    for p_new, p_old in zip(agent.policy.parameters(), old_agent.policy.parameters()):
        diff = (p_new - p_old).abs().max().item()
        max_diff = max(max_diff, diff)
    for v_new, v_old in zip(
        agent.value_net.parameters(), old_agent.value_net.parameters()
    ):
        diff = (v_new - v_old).abs().max().item()
        max_diff = max(max_diff, diff)

print(f"max param diff between agent and old_agent = {max_diff}")


def eval_vs_snapshot(
    game,
    agent,
    snapshot_agent,
    teammate_id: int,
    n_episodes: int = 50,
):
    """
    Evaluate current agent vs snapshot with side randomization.

    Team A: seats (0, teammate_id)
    Team B: the other two seats.

    For each game we flip which team is controlled by the current agent.
    """
    env_eval = rl_environment.Environment(game=game)

    wins_agent = 0
    losses_agent = 0
    draws = 0

    for ep in range(n_episodes):
        ts = env_eval.reset()

        # build team sets from this state
        st = env_eval.get_state
        inner = st._inner
        players = inner.players
        p0 = players[0]
        teammate = inner.teammate(p0)
        teammate_idx = players.index(teammate)
        assert teammate_idx == teammate_id, "teammate_id mismatch"

        teamA = {0, teammate_idx}
        teamB = set(range(len(players))) - teamA

        # flip: even eps -> agent is team A, odd eps -> agent is team B
        agent_on_teamA = ep % 2 == 0

        while not ts.last():
            p = ts.observations["current_player"]
            state = env_eval.get_state

            legal_ids = ts.observations["legal_actions"][p]
            if not legal_ids:
                break

            obs = np.array(ts.observations["info_state"][p], dtype=np.float32)
            legal_act_feats = [
                encode_action_features(state, p, aid) for aid in legal_ids
            ]

            # decide who acts for this seat
            if p in teamA:
                acting_agent = agent if agent_on_teamA else snapshot_agent
            else:
                acting_agent = snapshot_agent if agent_on_teamA else agent

            idx, chosen_feat, logp, value = acting_agent.select_action(
                obs, legal_act_feats, eval_mode=True
            )
            chosen_id = legal_ids[idx]

            ts = env_eval.step([chosen_id])

        # winner mapping
        state = env_eval.get_state
        inner = state._inner
        winner_team = inner.winner

        if winner_team is None:
            draws += 1
        else:
            # winner_team is a pair of player objects
            winner_seats = {players.index(p) for p in winner_team}
            teamA_won = winner_seats == teamA

            if agent_on_teamA:
                if teamA_won:
                    wins_agent += 1
                else:
                    losses_agent += 1
            else:
                if teamA_won:
                    losses_agent += 1
                else:
                    wins_agent += 1

    total = wins_agent + losses_agent + draws
    win_rate_all = wins_agent / total if total > 0 else 0.0
    non_draw = wins_agent + losses_agent
    win_rate_no_draws = wins_agent / non_draw if non_draw > 0 else 0.0

    return wins_agent, losses_agent, draws, win_rate_all, win_rate_no_draws


print("Initial eval vs snapshot BEFORE any training:")
ew, el, ed, wr_all, wr_no_draws = eval_vs_snapshot(
    game, agent, old_agent, teammate_id, n_episodes=EVAL_EPISODES
)
print(
    f"  wins={ew}, losses={el}, draws={ed}, "
    f"win_rate_all={wr_all:.3f}, win_rate_no_draws={wr_no_draws:.3f}"
)


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
        state_before = env.get_state
        inner_before = state_before._inner

        if current_player == 0 or current_player == teammate_id:
            # current agent players (team)
            obs = np.array(
                time_step.observations["info_state"][current_player], dtype=np.float32
            )

            fin_b, home_b, dist_b = team_progress(inner_before)

            legal_ids = time_step.observations["legal_actions"][current_player]
            state_for_actions = state_before
            legal_act_feats = [
                encode_action_features(state_for_actions, current_player, aid)
                for aid in legal_ids
            ]

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

            # env reward for this player
            env_reward = next_time_step.rewards[current_player]
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
            # snapshot opponents (both players on other team)
            obs = np.array(
                time_step.observations["info_state"][current_player], dtype=np.float32
            )
            legal_ids = time_step.observations["legal_actions"][current_player]
            if not legal_ids:
                break

            state_for_actions = state_before
            legal_act_feats = [
                encode_action_features(state_for_actions, current_player, aid)
                for aid in legal_ids
            ]

            idx, chosen_feat, logp, value = old_agent.select_action(
                obs, legal_act_feats, eval_mode=False
            )
            chosen_id = legal_ids[idx]
            time_step = env.step([chosen_id])

    # decide outcome from underlying game state
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

    # add outcome to final step reward for current agent
    if agent.current_episode:
        agent.current_episode[-1].reward += outcome
        agent.current_episode[-1].done = True

    # move episode into global buffer with GAE/returns
    agent.finish_episode()

    # PPO update every few episodes
    if (ep + 1) % UPDATE_EVERY_EP == 0:
        agent.update()

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

    # periodic evaluation vs snapshot and snapshot update
    if (ep + 1) % EVAL_EVERY_EP == 0:
        ew, el, ed, wr_all, wr_no_draws = eval_vs_snapshot(
            game, agent, old_agent, teammate_id, n_episodes=EVAL_EPISODES
        )
        print(
            f"[EVAL ep {ep+1}] vs snapshot: wins={ew}, losses={el}, draws={ed}, "
            f"win_rate_all={wr_all:.3f}, win_rate_no_draws={wr_no_draws:.3f}"
        )

        if wr_no_draws >= SNAPSHOT_THRESHOLD and (ew + el) > 0:
            print(
                f"[SNAPSHOT UPDATE] win_rate_no_draws={wr_no_draws:.3f} "
                f">= {SNAPSHOT_THRESHOLD:.2f}, updating snapshot."
            )

            # --- update snapshot weights ---
            old_agent.policy.load_state_dict(agent.policy.state_dict())
            old_agent.value_net.load_state_dict(agent.value_net.state_dict())

            # --- save snapshot to file ---
            SNAPSHOT_PATH = "dog_param_ppo_old.pt"
            torch.save(
                {
                    "policy": old_agent.policy.state_dict(),
                    "value_net": old_agent.value_net.state_dict(),
                },
                SNAPSHOT_PATH,
            )
            print(f"[SNAPSHOT SAVED] {SNAPSHOT_PATH}")


# final save
agent.save(MODEL_PATH)
print("Training finished, model saved.")
