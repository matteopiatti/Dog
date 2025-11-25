# eval_ppo_param.py

import os
import numpy as np

from DogGame import DogGame, NUM_ACTIONS
from open_spiel.python import rl_environment
from open_spiel.python.algorithms import random_agent

from parametric_ppo import ParametricPPOAgent
from action_features import encode_action_features, ACT_DIM


def eval_agent(env, agent, opponents, n_episodes=50):
    wins = 0
    losses = 0
    draws = 0
    total_env_reward_p0 = 0.0

    for _ in range(n_episodes):
        ts = env.reset()
        episode_env_reward_p0 = 0.0

        while not ts.last():
            p = ts.observations["current_player"]

            if p == 0:
                # observation for player 0
                obs = np.array(ts.observations["info_state"][0], dtype=np.float32)

                # underlying DogState to decode action ids
                state = env.get_state  # if get_state is a method, use env.get_state()

                legal_ids = ts.observations["legal_actions"][0]
                if not legal_ids:
                    # no legal move safeguard
                    break

                # build features for each legal action id
                legal_act_feats = [
                    encode_action_features(state, 0, aid) for aid in legal_ids
                ]

                # greedy (eval_mode=True) selection
                idx, chosen_feat, logp, value = agent.select_action(
                    obs, legal_act_feats, eval_mode=True
                )
                chosen_id = legal_ids[idx]

                ts = env.step([chosen_id])
            else:
                # random opponents as in training
                out = opponents[p - 1].step(ts, is_evaluation=True)
                ts = env.step([out.action])

            # accumulate raw env reward for player 0
            episode_env_reward_p0 += ts.rewards[0]

        # final underlying game state
        state = env.get_state
        inner = state._inner
        winner_team = inner.winner

        if winner_team is None:
            draws += 1
        elif inner.players[0] in winner_team:
            wins += 1
        else:
            losses += 1

        total_env_reward_p0 += episode_env_reward_p0

    total = wins + losses + draws
    win_rate_all = wins / total if total > 0 else 0.0
    non_draw = wins + losses
    win_rate_no_draws = wins / non_draw if non_draw > 0 else 0.0
    avg_env_reward_p0 = total_env_reward_p0 / total if total > 0 else 0.0

    return wins, losses, draws, win_rate_all, win_rate_no_draws, avg_env_reward_p0


def main():
    model_path = "dog_param_ppo.pt"
    if not os.path.exists(model_path):
        print(f"Model file '{model_path}' not found.")
        return

    game = DogGame()
    env = rl_environment.Environment(game=game)
    num_players = game.num_players()

    # infer observation size
    ts = env.reset()
    obs_dim = len(ts.observations["info_state"][0])

    # build PPO agent and load weights
    agent = ParametricPPOAgent(
        obs_dim=obs_dim,
        act_dim=ACT_DIM,
    )
    agent.load(model_path)
    print(f"Loaded model from {model_path}")

    # same random opponents as in training (players 1,2,3)
    opponents = [
        random_agent.RandomAgent(pid, NUM_ACTIONS, name=f"random_{pid}")
        for pid in range(1, num_players)
    ]

    n_eval = 50
    wins, losses, draws, wr_all, wr_no_draws, avg_r = eval_agent(
        env, agent, opponents, n_episodes=n_eval
    )

    print(f"Evaluated over {n_eval} games (greedy policy, no learning):")
    print(f"  wins      = {wins}")
    print(f"  losses    = {losses}")
    print(f"  draws     = {draws}")
    print(f"  win rate (including draws)      = {wr_all:.3f}")
    print(f"  win rate (excluding draws only) = {wr_no_draws:.3f}")
    print(f"  avg raw env reward for player 0 = {avg_r:.3f}")


if __name__ == "__main__":
    main()
