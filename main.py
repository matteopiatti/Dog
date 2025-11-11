from dog.engine import Engine
from dog.cli import render, select_action

def main():
  state = Engine.setup_game(4)
  # state = Engine.start_game(state)
  while not state.finished:
    render(state)
    # actions = legal_actions(state, state.current_player)
    action = select_action(state, ["do nothing"])
    Engine.step(state)

if __name__ == "__main__":
  main()