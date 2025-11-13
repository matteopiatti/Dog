## Quick orientation for AI coding agents

This repo is a small Python implementation of the board/card game "Dog".
Focus on the following short summary and concrete touchpoints before making changes.

Core architecture (big picture)

- Entry: `main.py` — calls `Engine.setup_game(...)` then `Engine.start_game(state)`.
- Game container: `dog/state.py::GameState` holds the mutable game snapshot (players, board, deck, turn state).
- Game flow: `dog/engine.py` — setup -> start_game -> start_round -> start_turn -> move_action. Most functions mutate `GameState` in-place and also return it.
- Rules and actions: `dog/move.py` generates legal `Action` objects and applies them via `move_action`.
- Board model: `dog/board.py` holds `track[]`, `home{}`, `start_fields` and `occupied_fields`; movement logic and capture are implemented here.

Important file-level patterns & conventions

- Dataclasses are used for plain data: `Player`, `GameState`, `Action`.
- `Marble` is a frozen dataclass with `eq=False`; marble identity (object identity) is used in sets and comparisons (e.g. `marbles_in_play`).
- Enums live in `dog/enums.py` and are the canonical source for card ranks, suits and player colors.
- CLI rendering: `dog/cli.py` prints colored output using ANSI escapes and expects interactive input prompts; many functions block on `input()`.
- Deck behavior: `dog/cards.py::Deck.deal` will reinitialize & shuffle the deck automatically when empty — tests or changes should account for this regeneration.

Mutability and calling style to watch for

- Most APIs mutate `GameState` and player objects in-place. Do not assume pure/functional updates.
- Several utilities are defined as plain functions inside classes (e.g. `Engine.start_turn` is defined on the class and called as `Engine.start_turn(state)`). Treat them as module-level helpers — they are not instance methods.

How to run locally (developer workflow)

- Requires Python 3.10+ (uses `X | None` union syntax). Run the game with:

```bash
python main.py
```

- The game is interactive (CLI); unit tests are not present. When making changes, run the game for a quick smoke test. Expect prompts and ANSI-colored output.

Concrete examples to reference when coding

- To inspect legal moves for the current player:
  - from `dog.move` import `generate_moves`
  - call `generate_moves(state, player, card)` — the function enumerates actions based on `card.rank` and `board.is_valid_move`.
- To apply an action: `dog.move.move_action(state, action)` — this updates the board and removes the played card from the player's hand.
- To start a marble on the board: `dog.board.Board.start_marble(player_idx, player)` — this uses `get_free_marble()` on `Player`.

Testing, safety notes and common pitfalls

- Watch for identity vs equality bugs: marbles are identified by object identity and stored in sets; do not attempt to create new Marble instances to refer to existing ones.
- CLI functions use `input()` and `print()` heavily; automated tests should mock input/print or refactor I/O into injectable interfaces.
- The code uses ANSI color escape codes embedded in strings (`enums.Colors`), so tests or snapshot assertions should strip color when comparing output.

If you make a behavior change that affects rules (move generation, capture, start logic), update `dog/move.py` and `dog/board.py` together — the logic is split between them.

Questions or missing context

- If rules are ambiguous (e.g., wrap-around board behavior, exact home/finish conditions), ask the maintainer for the intended rule set before changing core move logic.

If anything in this summary is unclear or you'd like more examples (ID lookups, sample GameState dumps), tell me which area to expand and I'll update this file.
