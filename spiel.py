import pyspiel
import numpy as np
from DogGame import DogGame
from dog.cli import render
import random

game = DogGame()
state = game.new_initial_state()
while not state.is_terminal():
    acts = state.legal_actions()
    if not acts:
        break
    a = random.choice(acts)
    state.apply_action(a)
    render(state._inner)
print("Returns:", state.returns())
