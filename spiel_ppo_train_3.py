# train_ppo_param.py
# Self-play training: agent team vs league of frozen snapshot teams.
# Latest snapshot is always updated to current agent.
# League accumulates older snapshots periodically.
# Progress is also measured vs fixed random baseline.

import os
import random
import numpy as np
import pyspiel
import torch

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

EVAL_EVERY_EP = 100  # how often to evaluate vs latest snapshot
EVAL_EPISODES = 50  # episodes per eval
EVAL_RANDOM_EVERY_EP = 1000  # how often to evaluate vs random baseline

# PPO hyperparameters (must match between agent and snapshots)
PPO_LR = 1e-4
PPO_CLIP_EPS = 0.1
PPO_ENTROPY_COEF = 0.02
PPO_UPDATE_EPOCHS = 6
PPO_MINIBATCH_SIZE = 512

# snapshot league settings
MAX_LEAGUE_SIZE = 5
SNAPSHOT_ADD_PERIOD = 5  # add a frozen snapshot to league every N evals

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
    lr=PPO_LR,
    clip_eps=PPO_CLIP_EPS,
    entropy_coef=PPO_ENTROPY_COEF,
    update_epochs=PPO_UPDATE_EPOCHS,
    minibatch_size=PPO_MINIBATCH_SIZE,
)

if os.path.exists(MODEL_PATH):
    try:
        agent.load(MODEL_PATH)
        print("Loaded existing PPO checkpoint.")
    except RuntimeError as e:
        print("WARNING: could not load checkpoint (likely architecture change).")
        print("Details:", e)
        print("Starting fresh with new architecture.")
else:
    print("No existing checkpoint, starting fresh.")

# latest snapshot opponent
old_agent = ParametricPPOAgent(
    obs_dim=obs_dim,
    act_dim=ACT_DIM,
    lr=PPO_LR,
    clip_eps=PPO_CLIP_EPS,
    entropy_coef=PPO_ENTROPY_COEF,
    update_epochs=PPO_UPDATE_EPOCHS,
    minibatch_size=PPO_MINIBATCH_SIZE,
)
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

# league of snapshot opponents (start with one)
snapshot_league = [old_agent]
eval_counter = 0

# --------------------------------------------------
# helpers
# --------------------------------------------------


def team_progress(inner, team_indices) -> tuple[float, float, float]:
    """
    Progress for a given team (by seat indices).
    """
    board = inner.board
    players = inner.players
    team_players = [players[i] for i in team_indices]

    finished = float(board.team_finished_marbles(tuple(team_players)))

    home_count = 0.0
    for p in team_players:
        home_count += sum(1 for m in board.home[p] if m is not None)

    distance = float(
        board.total_distance_to_home(team_players[0])
        + board.total_distance_to_home(team_players[1])
    )

    return finished, home_count, distance


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

        st = env_eval.get_state
        inner = st._inner
        players = inner.players
        p0 = players[0]
        teammate = inner.teammate(p0)
        teammate_idx = players.index(teammate)
        assert teammate_idx == teammate_id, "teammate_id mismatch"

        teamA = {0, teammate_idx}
        teamB = set(range(len(players))) - teamA

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

            if p in teamA:
                acting_agent = agent if agent_on_teamA else snapshot_agent
            else:
                acting_agent = snapshot_agent if agent_on_teamA else agent

            idx, chosen_feat, logp, value = acting_agent.select_action(
                obs, legal_act_feats, eval_mode=True
            )
            chosen_id = legal_ids[idx]

            ts = env_eval.step([chosen_id])

        state = env_eval.get_state
        inner = state._inner
        winner_team = inner.winner

        if winner_team is None:
            draws += 1
        else:
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


def eval_vs_random(game, agent, teammate_id: int, n_episodes: int = 50):
    """
    Evaluate agent team vs random opponents.
    Team = {0, teammate_id}; others are random.
    """
    env_eval = rl_environment.Environment(game=game)
    num_players = game.num_players()

    rand_agents = [
        random_agent.RandomAgent(pid, NUM_ACTIONS, name=f"random_{pid}")
        for pid in range(num_players)
    ]

    wins = 0
    losses = 0
    draws = 0

    for _ in range(n_episodes):
        ts = env_eval.reset()
        state = env_eval.get_state
        inner = state._inner
        players = inner.players
        p0 = players[0]
        teammate = inner.teammate(p0)
        teammate_idx = players.index(teammate)
        teamA = {0, teammate_idx}

        while not ts.last():
            p = ts.observations["current_player"]
            state = env_eval.get_state

            legal_ids = ts.observations["legal_actions"][p]
            if not legal_ids:
                break

            if p in teamA:
                obs = np.array(ts.observations["info_state"][p], dtype=np.float32)
                legal_act_feats = [
                    encode_action_features(state, p, aid) for aid in legal_ids
                ]
                idx, chosen_feat, logp, value = agent.select_action(
                    obs, legal_act_feats, eval_mode=True
                )
                chosen_id = legal_ids[idx]
            else:
                out = rand_agents[p].step(ts, is_evaluation=True)
                chosen_id = out.action

            ts = env_eval.step([chosen_id])

        state = env_eval.get_state
        inner = state._inner
        winner_team = inner.winner

        if winner_team is None:
            draws += 1
        else:
            winner_seats = {players.index(pl) for pl in winner_team}
            if winner_seats == teamA:
                wins += 1
            else:
                losses += 1

    total = wins + losses + draws
    wr_all = wins / total if total > 0 else 0.0
    non_draw = wins + losses
    wr_no_draws = wins / non_draw if non_draw > 0 else 0.0
    return wins, losses, draws, wr_all, wr_no_draws


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
# training loop
# --------------------------------------------------

for ep in range(NUM_EPISODES):
    time_step = env.reset()
    agent.current_episode = []  # ensure clean

    # build team sets for this episode
    state_ep = env.get_state
    inner_ep = state_ep._inner
    players = inner_ep.players
    p0 = players[0]
    teammate = inner_ep.teammate(p0)
    teammate_idx = players.index(teammate)
    teamA = {0, teammate_idx}
    teamB = set(range(len(players))) - teamA

    # flip which team the agent controls this episode
    agent_on_teamA = ep % 2 == 0
    agent_team = teamA if agent_on_teamA else teamB
    snapshot_team = teamB if agent_on_teamA else teamA

    # sample opponent for this episode from league
    opponent_agent = random.choice(snapshot_league)

    while not time_step.last():
        current_player = time_step.observations["current_player"]
        state_before = env.get_state
        inner_before = state_before._inner

        if current_player in agent_team:
            # agent-controlled players (for this episode)
            obs = np.array(
                time_step.observations["info_state"][current_player], dtype=np.float32
            )

            legal_ids = time_step.observations["legal_actions"][current_player]
            state_for_actions = state_before
            legal_act_feats = [
                encode_action_features(state_for_actions, current_player, aid)
                for aid in legal_ids
            ]

            # progress BEFORE
            fin_b, home_b, dist_b = team_progress(inner_before, agent_team)

            idx, chosen_feat, logp, value = agent.select_action(
                obs, legal_act_feats, eval_mode=False
            )
            chosen_id = legal_ids[idx]

            next_time_step = env.step([chosen_id])

            state_after = env.get_state
            inner_after = state_after._inner

            # progress AFTER
            fin_a, home_a, dist_a = team_progress(inner_after, agent_team)
            df = fin_a - fin_b
            dh = home_a - home_b
            dd = dist_b - dist_a  # positive if closer

            NUM_FIELDS = inner_after.board.NUM_FIELDS
            norm_dist = NUM_FIELDS * 8.0

            # small progress shaping
            progress_reward = 0.02 * df + 0.01 * dh + 0.02 * (dd / norm_dist)

            # env reward for this player
            env_reward = next_time_step.rewards[current_player]
            r = env_reward + progress_reward
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
            # snapshot-opponent-controlled players
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

            idx, chosen_feat, logp, value = opponent_agent.select_action(
                obs, legal_act_feats, eval_mode=False
            )
            chosen_id = legal_ids[idx]
            time_step = env.step([chosen_id])

    # outcome relative to agent_team
    state = env.get_state
    inner = state._inner
    winner_team = inner.winner

    if winner_team is None:
        outcome = 0.0
        draws += 1
    else:
        win_seats = {players.index(p) for p in winner_team}
        if win_seats == agent_team:
            outcome = 1.0
            wins += 1
        else:
            outcome = -1.0
            losses += 1

    # strong terminal signal
    if agent.current_episode:
        agent.current_episode[-1].reward += 5.0 * outcome
        agent.current_episode[-1].done = True

    # move episode into global buffer with GAE/returns
    agent.finish_episode()

    # PPO update every few episodes
    if (ep + 1) % UPDATE_EVERY_EP == 0:
        agent.update()

    # periodic logging and checkpoint (agent_team winrate)
    if (ep + 1) % LOG_EVERY_EP == 0:
        total = wins + losses + draws
        win_rate = wins / total if total > 0 else 0.0
        print(
            f"Episode {ep+1}: agent_team_wins={wins}, losses={losses}, draws={draws}, "
            f"win_rate_agent_team={win_rate:.3f}"
        )
        wins = 0
        losses = 0
        draws = 0
        agent.save(MODEL_PATH)

    # periodic evaluation vs latest snapshot and league update
    if (ep + 1) % EVAL_EVERY_EP == 0:
        ew, el, ed, wr_all, wr_no_draws = eval_vs_snapshot(
            game, agent, old_agent, teammate_id, n_episodes=EVAL_EPISODES
        )
        print(
            f"[EVAL ep {ep+1}] vs latest snapshot: wins={ew}, losses={el}, draws={ed}, "
            f"win_rate_all={wr_all:.3f}, win_rate_no_draws={wr_no_draws:.3f}"
        )

        eval_counter += 1

        # always update latest snapshot to current agent
        old_agent.policy.load_state_dict(agent.policy.state_dict())
        old_agent.value_net.load_state_dict(agent.value_net.state_dict())

        # periodically add frozen snapshot to league
        if eval_counter % SNAPSHOT_ADD_PERIOD == 0:
            print("[LEAGUE] Adding new snapshot to league.")
            new_snapshot = ParametricPPOAgent(
                obs_dim=obs_dim,
                act_dim=ACT_DIM,
                lr=PPO_LR,
                clip_eps=PPO_CLIP_EPS,
                entropy_coef=PPO_ENTROPY_COEF,
                update_epochs=PPO_UPDATE_EPOCHS,
                minibatch_size=PPO_MINIBATCH_SIZE,
            )
            new_snapshot.policy.load_state_dict(agent.policy.state_dict())
            new_snapshot.value_net.load_state_dict(agent.value_net.state_dict())
            snapshot_league.append(new_snapshot)
            if len(snapshot_league) > MAX_LEAGUE_SIZE:
                snapshot_league.pop(0)

            SNAPSHOT_PATH = "dog_param_ppo_old.pt"
            torch.save(
                {
                    "policy": old_agent.policy.state_dict(),
                    "value_net": old_agent.value_net.state_dict(),
                },
                SNAPSHOT_PATH,
            )
            print(f"[SNAPSHOT SAVED] {SNAPSHOT_PATH}")

    # periodic evaluation vs random baseline
    if (ep + 1) % EVAL_RANDOM_EVERY_EP == 0:
        rw, rl, rd, r_all, r_no = eval_vs_random(
            game, agent, teammate_id, n_episodes=EVAL_EPISODES
        )
        print(
            f"[EVAL-RANDOM ep {ep+1}] vs random: wins={rw}, losses={rl}, draws={rd}, "
            f"win_rate_all={r_all:.3f}, win_rate_no_draws={r_no:.3f}"
        )

# final save
agent.save(MODEL_PATH)
print("Training finished, model saved.")
