# play_vs_bots_cli.py
#
# You (human, player 0) + random teammate
# vs two PPO bots.
#
# Uses OpenSpiel rl_environment + dog.cli.render(inner_state)
# for visualization.

import os
import numpy as np

from DogGame import DogGame, NUM_ACTIONS
from dog.cli import render, print_no_actions as cli_print_no_actions

from open_spiel.python import rl_environment
from open_spiel.python.algorithms import random_agent

from parametric_ppo import ParametricPPOAgent
from action_features import encode_action_features, ACT_DIM

MODEL_PATH = "dog_param_ppo.pt"


def human_choose_action(env, ts, player_id: int) -> int:
    """Let human choose one of the legal OpenSpiel actions for this player."""
    state = env.get_state
    inner = state._inner

    legal_ids = ts.observations["legal_actions"][player_id]
    if not legal_ids:
        cli_print_no_actions()
        return -1

    print("\n=== Current game state ===")
    render(inner)
    print("==========================")
    print(f"Your player id: {player_id}")
    print("Legal moves (index: action_id):")
    for i, aid in enumerate(legal_ids):
        print(f"  {i+1}: action_id={aid}")

    while True:
        choice = input("Choose move index (or 'q' to quit this game): ").strip()
        if choice.lower() in ["q", "quit", "exit"]:
            return -1
        try:
            idx = int(choice)
            if 0 <= idx <= len(legal_ids):
                return legal_ids[idx - 1]
            print(f"Invalid index, please enter 1..{len(legal_ids)}")
        except ValueError:
            print("Please enter a valid integer index or 'q' to quit.")


def ppo_choose_action(agent, env, ts, player_id: int) -> int:
    """Choose greedy PPO action for given OpenSpiel timestep."""
    state = env.get_state
    legal_ids = ts.observations["legal_actions"][player_id]
    if not legal_ids:
        return -1

    obs = np.array(ts.observations["info_state"][player_id], dtype=np.float32)
    legal_act_feats = [
        encode_action_features(state, player_id, aid) for aid in legal_ids
    ]
    idx, chosen_feat, logp, value = agent.select_action(
        obs, legal_act_feats, eval_mode=True
    )
    return legal_ids[idx]


def play_one_game(env, agent, teammate_random, teammate_id: int) -> None:
    """Play a single episode: human+random vs 2 PPO bots."""
    ts = env.reset()
    state = env.get_state
    inner = state._inner

    players = inner.players
    p0 = players[0]
    teammate_player = inner.teammate(p0)
    assert teammate_id == players.index(teammate_player)

    opponent_ids = [pid for pid in range(len(players)) if pid not in (0, teammate_id)]

    print("\nNew game started.")
    print(f"You are player 0. Your teammate is player {teammate_id}.")
    print(f"Opponents: players {opponent_ids}\n")

    while not ts.last():
        p = ts.observations["current_player"]
        state = env.get_state
        inner = state._inner

        if p == 0:
            chosen_id = human_choose_action(env, ts, p)
            if chosen_id < 0:
                print("You chose to quit this game.")
                return
            ts = env.step([chosen_id])

        elif p == teammate_id:
            out = teammate_random.step(ts, is_evaluation=True)
            ts = env.step([out.action])

        else:
            chosen_id = ppo_choose_action(agent, env, ts, p)
            if chosen_id < 0:
                cli_print_no_actions()
                return
            ts = env.step([chosen_id])

    # episode finished
    state = env.get_state
    inner = state._inner

    print("\n=== Final game state ===")
    render(inner)
    print("========================")

    winner_team = inner.winner

    if winner_team is None:
        print("Result: draw.")
    else:
        win_seats = [inner.players.index(pl) for pl in winner_team]
        print(f"Winning team seats: {win_seats}")
        if 0 in win_seats:
            print("You (and your teammate) won!")
        else:
            print("The bots won.")


def main():
    if not os.path.exists(MODEL_PATH):
        print(f"Model file '{MODEL_PATH}' not found. Train first.")
        return

    # OpenSpiel environment
    game = DogGame()
    env = rl_environment.Environment(game=game)

    # infer obs dim from OpenSpiel
    ts = env.reset()
    obs_dim = len(ts.observations["info_state"][0])

    # PPO bot
    agent = ParametricPPOAgent(
        obs_dim=obs_dim,
        act_dim=ACT_DIM,
    )
    agent.load(MODEL_PATH)
    print(f"Loaded PPO model from {MODEL_PATH}")

    # determine teammate id for player 0
    state = env.get_state
    inner = state._inner
    players = inner.players
    p0 = players[0]
    teammate_player = inner.teammate(p0)
    teammate_id = players.index(teammate_player)
    print(f"Your teammate will be seat {teammate_id}")

    # random teammate agent (OpenSpiel)
    teammate_random = random_agent.RandomAgent(
        teammate_id, NUM_ACTIONS, name=f"random_{teammate_id}"
    )

    # Game loop
    while True:
        play_one_game(env, agent, teammate_random, teammate_id)
        again = input("\nPlay another game? [y/N]: ").strip().lower()
        if again not in ["y", "yes"]:
            break


if __name__ == "__main__":
    main()
