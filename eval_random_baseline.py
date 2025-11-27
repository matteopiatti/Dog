# eval_random_baseline.py
#
# Run N games of pure random play and measure how often
# the TEAM CONTAINING PLAYER 0 wins.

import numpy as np

from DogGame import DogGame, NUM_ACTIONS
from open_spiel.python import rl_environment
from open_spiel.python.algorithms import random_agent


NUM_EPISODES = 10_000


def main():
    game = DogGame()
    env = rl_environment.Environment(game=game)
    num_players = game.num_players()

    # 4 random agents for all seats
    agents = [
        random_agent.RandomAgent(pid, NUM_ACTIONS, name=f"random_{pid}")
        for pid in range(num_players)
    ]

    team0_wins = 0
    other_team_wins = 0
    draws = 0

    for ep in range(NUM_EPISODES):
        ts = env.reset()

        # play one episode
        while not ts.last():
            current_player = ts.observations["current_player"]
            agent = agents[current_player]
            out = agent.step(ts, is_evaluation=True)
            ts = env.step([out.action])

        # episode finished, inspect underlying game state
        state = env.get_state
        inner = state._inner

        winner_team = inner.winner  # None or tuple/list of 2 player objects

        if winner_team is None:
            draws += 1
        else:
            # team containing player 0?
            p0 = inner.players[0]
            if p0 in winner_team:
                team0_wins += 1
            else:
                other_team_wins += 1

        if (ep + 1) % 1000 == 0:
            print(f"Finished {ep+1} episodes...")

    total = team0_wins + other_team_wins + draws

    print("\n=== Random vs Random baseline over " f"{NUM_EPISODES} games ===")
    print(f"Team containing player 0 wins : {team0_wins}")
    print(f"Other team wins              : {other_team_wins}")
    print(f"Draws                        : {draws}")
    print(f"Total                        : {total}")

    win_rate_team0_all = team0_wins / total if total > 0 else 0.0
    non_draw = team0_wins + other_team_wins
    win_rate_team0_no_draws = team0_wins / non_draw if non_draw > 0 else 0.0

    print(f"Win rate team0 (including draws)      = {win_rate_team0_all:.3f}")
    print(f"Win rate team0 (excluding draws only) = {win_rate_team0_no_draws:.3f}")


if __name__ == "__main__":
    main()
