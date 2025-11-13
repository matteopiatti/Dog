from dog.engine import setup_game, step
from dog.cli import render, select_action

def main():
  state = setup_game(4)
  # state = Engine.start_game(state)
  while not state.finished:
    step(state)
    render(state)
    # print('Available actions:', actions)
    # action = select_action(state, actions)
    # Engine.step(state)

if __name__ == "__main__":
  main()