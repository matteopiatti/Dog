import pyspiel
import numpy as np

game = pyspiel.load_game("tic_tac_toe")
state = game.new_initial_state()

step = 0
while not state.is_terminal():
    print("Step", step)
    print(state)  # built-in string representation of the board
    player = state.current_player()
    legal = state.legal_actions()
    action = int(np.random.choice(legal))
    print("Player", player, "plays", state.action_to_string(player, action))
    print("-" * 30)
    state.apply_action(action)
    step += 1

print("Final state:")
print(state)
print("Returns:", state.returns())
