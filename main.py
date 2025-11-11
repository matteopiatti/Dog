from dog.engine import Engine
from dog.cli import clear_screen

def main():
  state = Engine.setup_game(4)
  state = Engine.start_game(state)

if __name__ == "__main__":
  main()