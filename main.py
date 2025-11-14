from dog.engine import setup_game, step
from dog.cli import render

def main():
  state = setup_game(4)
  while not state.finished:
    step(state)
    render(state)

if __name__ == "__main__":
  main()