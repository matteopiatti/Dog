from agent.parametricagent import ParametricAgent
from dog.engine import setup_game, step
from dog.cli import render
from pathlib import Path
from agent.humanagent import HumanAgent
from agent.randomagent import RandomAgent
from agent.trainingagent import TrainingAgent
from train import train, POOL_SIZE, load_pool
import json

""" 
TODO:
- Rewrite Train function to check for speed of random games
- implement a complete state copying function to determine future moves on temp states
- different weight vectors for action and card switch
- play against random agent during training for more diversity
 """


def main():
    choice = input("Training or playing? Type 't' or 'p': ").strip().lower()
    if choice == "p":
        pool, fitness = load_pool()
        best_idx = max(range(POOL_SIZE), key=lambda i: fitness[i])
        worst_idx = min(range(POOL_SIZE), key=lambda i: fitness[i])
        best = pool[best_idx]
        worst = pool[worst_idx]
        agents = [
            HumanAgent(),
            TrainingAgent(best["action"], best["switch"]),
            RandomAgent(),
            TrainingAgent(worst["action"], worst["switch"]),
        ]
        state = setup_game(num_players=4, agents=agents)
        while not state.finished:
            render(state)
            step(state)
            # sleep(0.5)

        if state.finished:
            render(state)
            print(f"Game over! The winning team is: {state.winner}")
            # save_state(state, "final_state.json")
    elif choice == "t":
        best_weights = train(iterations=1000)
        print(f"Best weights after training: {best_weights}")
        # save best weights to file
        Path("best_weights.json").write_text(
            json.dumps(best_weights, indent=2), encoding="utf-8"
        )

    # state = (
    #     load_state("savegame.json")
    #     if Path("savegame.json").exists()
    #     else setup_game(num_players=4)
    # )
    # try:
    #     while not state.finished:
    #         render(state)
    #         step(state)
    # except KeyboardInterrupt:
    #     print("\nGame interrupted. Saving state...")
    #     save_state(state)
    #     print("State saved to 'savegame.json'.")


if __name__ == "__main__":
    main()
