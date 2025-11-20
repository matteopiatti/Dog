# eval_dqn.py  (parametric DQN, players 0+2 vs random 1+3)

import os
import numpy as np

from DogGame import DogGame, NUM_ACTIONS
from open_spiel.python import rl_environment
from open_spiel.python.algorithms import random_agent

from dqn_parametric import ParametricDQNAgent
from action_features import encode_action_features, ACT_DIM


def eval_agent(env, agent, opponents, n_episodes=50):
    wins = 0
    losses = 0
    draws = 0

    for _ in range(n_episodes):
        ts = env.reset()

        while not ts.last():
            p = ts.observations["current_player"]

            if p in (0, 2):
                # observation for current DQN player (0 or 2)
                obs = np.array(ts.observations["info_state"][p], dtype=np.float32)

                # underlying DogState to decode action ids
                state = env.get_state

                legal_ids = ts.observations["legal_actions"][p]
                if not legal_ids:
                    # no legal move (shouldn't happen often, but safeguard)
                    break

                # build features for each legal action id from this player's POV
                legal_act_feats = [
                    encode_action_features(state, p, aid) for aid in legal_ids
                ]

                # greedy (eval_mode=True) selection
                idx, chosen_feat = agent.select_action(
                    obs, legal_act_feats, eval_mode=True
                )
                chosen_id = legal_ids[idx]

                ts = env.step([chosen_id])

            else:
                # random opponents (players 1 and 3)
                out = opponents[p].step(ts, is_evaluation=True)
                ts = env.step([out.action])

        # decide win/loss/draw from underlying game state
        state = env.get_state
        inner = state._inner
        winner_team = inner.winner  # tuple(Player, Player) or None
        players = inner.players

        # our team is players[0] and players[2]
        if winner_team is None:
            draws += 1
        elif players[0] in winner_team or players[2] in winner_team:
            wins += 1
        else:
            losses += 1

    total = wins + losses + draws
    win_rate_all = wins / total if total > 0 else 0.0
    non_draw = wins + losses
    win_rate_no_draws = wins / non_draw if non_draw > 0 else 0.0

    return wins, losses, draws, win_rate_all, win_rate_no_draws


def main():
    model_path = "dog_dqn_param.pt"
    if not os.path.exists(model_path):
        print(f"Model file '{model_path}' not found.")
        return

    game = DogGame()
    env = rl_environment.Environment(game=game)
    num_players = game.num_players()

    # infer observation size
    ts = env.reset()
    obs_dim = len(ts.observations["info_state"][0])

    # build parametric DQN agent and load weights
    agent = ParametricDQNAgent(
        player_id=0,  # id not important for eval here
        obs_dim=obs_dim,
        act_dim=ACT_DIM,
    )
    agent.load(model_path)
    print(f"Loaded model from {model_path}")

    # random opponents for players 1 and 3 only
    opponents = {
        1: random_agent.RandomAgent(1, NUM_ACTIONS, name="random_1"),
        3: random_agent.RandomAgent(3, NUM_ACTIONS, name="random_3"),
    }

    n_eval = 50
    wins, losses, draws, wr_all, wr_no_draws = eval_agent(
        env, agent, opponents, n_episodes=n_eval
    )

    print(
        f"Evaluated over {n_eval} games (players 0+2 = DQN, 1+3 = random, greedy policy, no learning):"
    )
    print(f"  wins  = {wins}")
    print(f"  losses= {losses}")
    print(f"  draws = {draws}")
    print(f"  win rate (including draws)      = {wr_all:.3f}")
    print(f"  win rate (excluding draws only) = {wr_no_draws:.3f}")


if __name__ == "__main__":
    main()
