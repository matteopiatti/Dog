import os
import json
import random
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from agent.parametricagent import ParametricAgent
from agent.randomagent import RandomAgent
from agent.trainingagent import TrainingAgent
from dog.engine import setup_game, step

FEATURE_DIM = 9
POOL_SIZE = 20
SAVE_FILE = "weights.json"


def train(iterations=500):
    pool, fitness = load_pool()

    for it in range(iterations):
        parent_idx = random.randrange(POOL_SIZE)
        parent = pool[parent_idx]
        mutated_w = mutate(parent, sigma=0.1)

        opponent_indices = random.sample(range(POOL_SIZE), k=min(3, POOL_SIZE))
        winrates = []
        for oi in opponent_indices:
            wr = evaluate_weights(mutated_w, pool[oi], games=60)
            winrates.append(wr)

        fitness_new = sum(winrates) / len(winrates)
        print(f"iter {it}, fitness_new={fitness_new:.3f}")

        worst_idx = min(range(POOL_SIZE), key=lambda i: fitness[i])
        if fitness_new > fitness[worst_idx] + 0.03:
            pool[worst_idx] = mutated_w
            fitness[worst_idx] = fitness_new
            print(f"New best! Replaced index {worst_idx}")
        save_pool(pool, fitness)
    best_idx = max(range(POOL_SIZE), key=lambda i: fitness[i])
    return pool[best_idx]


def evaluate_weights(w_candidate, w_opponent, games=72):
    args = [(w_candidate, w_opponent, g) for g in range(games)]
    with ProcessPoolExecutor() as ex:
        results = list(ex.map(play_single_game, args))
    print(results)
    return sum(results) / games


def play_single_game(args):
    w_candidate, w_opponent, g = args

    if g % 2 == 0:
        bots = [
            TrainingAgent(w_candidate["action"], w_candidate["switch"]),
            TrainingAgent(w_opponent["action"], w_opponent["switch"]),
            TrainingAgent(w_candidate["action"], w_candidate["switch"]),
            TrainingAgent(w_opponent["action"], w_opponent["switch"]),
        ]
    else:
        bots = [
            TrainingAgent(w_candidate["action"], w_candidate["switch"]),
            RandomAgent(),
            TrainingAgent(w_candidate["action"], w_candidate["switch"]),
            RandomAgent(),
        ]

    state = setup_game(num_players=4, agents=bots)

    while not state.finished:
        step(state)

    return state.winner == (
        state.players[0],
        state.players[2],
    )


def load_pool():
    if os.path.exists(SAVE_FILE):
        data = json.loads(Path(SAVE_FILE).read_text())
        return data["pool"], data["fitness"]
    else:
        # initialize new pool
        pool = [
            {
                "action": random_weights(),
                "switch": random_weights(),
            }
            for _ in range(POOL_SIZE)
        ]
        fitness = [0.5] * POOL_SIZE
        return pool, fitness


def save_pool(pool, fitness):
    data = {"pool": pool, "fitness": fitness}
    Path(SAVE_FILE).write_text(json.dumps(data, indent=2))


def random_weights():
    return [random.uniform(-1, 1) for _ in range(FEATURE_DIM)]


def mutate(w, sigma=0.2):
    return {
        "action": [wi + random.gauss(0, sigma) for wi in w["action"]],
        "switch": [wi + random.gauss(0, sigma) for wi in w["switch"]],
    }
