# eval_dqn.py
import os
import numpy as np

from DogGame import DogGame, NUM_ACTIONS
from open_spiel.python import rl_environment
from open_spiel.python.algorithms import random_agent

from dqn_agent import DQNAgent


def eval_agent(env, dqn, opponents, n_episodes=50):
    wins = 0
    losses = 0
    draws = 0

    for _ in range(n_episodes):
        ts = env.reset()

        while not ts.last():
            p = ts.observations["current_player"]
            if p == 0:
                obs = np.array(ts.observations["info_state"][0], dtype=np.float32)
                legal_actions = ts.observations["legal_actions"][0]
                action = dqn.select_action(obs, legal_actions, eval_mode=True)
                ts = env.step([action])
            else:
                out = opponents[p - 1].step(ts, is_evaluation=True)
                ts = env.step([out.action])

        # decide win/loss/draw from underlying game state
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
    model_path = "dog_dqn.pt"
    if not os.path.exists(model_path):
        print(f"Model file '{model_path}' not found.")
        return

    game = DogGame()
    env = rl_environment.Environment(game=game)
    num_players = game.num_players()

    # infer observation size
    ts = env.reset()
    obs_dim = len(ts.observations["info_state"][0])

    # build DQN agent and load weights
    dqn = DQNAgent(
        player_id=0,
        obs_dim=obs_dim,
        num_actions=NUM_ACTIONS,
    )
    dqn.load(model_path)
    print(f"Loaded model from {model_path}")

    # same random opponents as in training (players 1,2,3)
    opponents = [
        random_agent.RandomAgent(pid, NUM_ACTIONS, name=f"random_{pid}")
        for pid in range(1, num_players)
    ]

    n_eval = 50
    wins, losses, draws, wr_all, wr_no_draws = eval_agent(
        env, dqn, opponents, n_episodes=n_eval
    )

    print(f"Evaluated over {n_eval} games:")
    print(f"  wins  = {wins}")
    print(f"  losses= {losses}")
    print(f"  draws = {draws}")
    print(f"  win rate (including draws)      = {wr_all:.3f}")
    print(f"  win rate (excluding draws only) = {wr_no_draws:.3f}")


if __name__ == "__main__":
    main()
