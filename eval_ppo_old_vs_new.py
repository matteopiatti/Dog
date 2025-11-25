# eval_ppo_vs_old.py
# Evaluate: team (current model) vs team (old model snapshot)

import os
import numpy as np

from DogGame import DogGame, NUM_ACTIONS
from open_spiel.python import rl_environment

from parametric_ppo import ParametricPPOAgent
from action_features import encode_action_features, ACT_DIM


GOOD_MODEL_PATH = "dog_param_ppo.pt"
OLD_MODEL_PATH = "dog_param_ppo_old.pt"
N_EVAL_EPISODES = 100


def eval_teams(env, game, good_agent, old_agent, teammate_id, n_episodes=50):
    wins = 0
    losses = 0
    draws = 0

    for _ in range(n_episodes):
        ts = env.reset()

        while not ts.last():
            p = ts.observations["current_player"]
            state = env.get_state

            legal_ids = ts.observations["legal_actions"][p]
            if not legal_ids:
                break

            obs = np.array(ts.observations["info_state"][p], dtype=np.float32)
            legal_act_feats = [
                encode_action_features(state, p, aid) for aid in legal_ids
            ]

            if p == 0 or p == teammate_id:
                idx, _, _, _ = good_agent.select_action(
                    obs, legal_act_feats, eval_mode=True
                )
            else:
                idx, _, _, _ = old_agent.select_action(
                    obs, legal_act_feats, eval_mode=True
                )

            chosen_id = legal_ids[idx]
            ts = env.step([chosen_id])

        state = env.get_state
        inner = state._inner
        winner_team = inner.winner

        if winner_team is None:
            draws += 1
        elif inner.players[0] in winner_team:
            wins += 1
        else:
            losses += 1

    total = wins + losses + draws
    win_rate_all = wins / total if total > 0 else 0.0
    non_draw = wins + losses
    win_rate_no_draws = wins / non_draw if non_draw > 0 else 0.0

    return wins, losses, draws, win_rate_all, win_rate_no_draws


def main():
    if not os.path.exists(GOOD_MODEL_PATH):
        print(f"Good model file '{GOOD_MODEL_PATH}' not found.")
        return
    if not os.path.exists(OLD_MODEL_PATH):
        print(f"Old model file '{OLD_MODEL_PATH}' not found.")
        return

    game = DogGame()
    env = rl_environment.Environment(game=game)
    num_players = game.num_players()

    ts = env.reset()
    obs_dim = len(ts.observations["info_state"][0])

    # determine teammate id of player 0
    state0 = env.get_state
    inner0 = state0._inner
    p0 = inner0.players[0]
    teammate_player = inner0.teammate(p0)
    teammate_id = inner0.players.index(teammate_player)
    print(f"Player 0 teammate id: {teammate_id}")

    # build agents
    good_agent = ParametricPPOAgent(
        obs_dim=obs_dim,
        act_dim=ACT_DIM,
    )
    old_agent = ParametricPPOAgent(
        obs_dim=obs_dim,
        act_dim=ACT_DIM,
    )

    good_agent.load(GOOD_MODEL_PATH)
    old_agent.load(OLD_MODEL_PATH)
    print(f"Loaded good model from {GOOD_MODEL_PATH}")
    print(f"Loaded old model from {OLD_MODEL_PATH}")

    wins, losses, draws, wr_all, wr_no_draws = eval_teams(
        env,
        game,
        good_agent,
        old_agent,
        teammate_id,
        n_episodes=N_EVAL_EPISODES,
    )

    print(
        f"Evaluated over {N_EVAL_EPISODES} games: "
        f"team(good: players 0,{teammate_id}) vs team(old)."
    )
    print(f"  wins      = {wins}")
    print(f"  losses    = {losses}")
    print(f"  draws     = {draws}")
    print(f"  win rate (including draws)      = {wr_all:.3f}")
    print(f"  win rate (excluding draws only) = {wr_no_draws:.3f}")


if __name__ == "__main__":
    main()
