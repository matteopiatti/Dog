def card_to_dict(card: Card) -> dict:
    return {
        "rank": card.rank.name,
        "suit": card.suit.name,
    }


def card_from_dict(d: dict) -> Card:
    rank = CardType[d["rank"]]
    suit = CardSuit[d["suit"]]
    return Card(rank, suit)


def state_to_dict(state: GameState) -> dict:
    # players
    players = []
    for p in state.players:
        players.append(
            {
                "name": p.name,
                "color": p.color.name,  # Colors enum
            }
        )

    # helper to map objects -> indices
    player_index = {p: i for i, p in enumerate(state.players)}
    marble_index = {}
    for pi, p in enumerate(state.players):
        for mi, m in enumerate(p.marbles):
            marble_index[m] = (pi, mi)

    # track
    track = []
    for m, p in state.board.track:
        if m is None or p is None:
            track.append(None)
        else:
            pi, mi = marble_index[m]
            track.append({"player": pi, "marble": mi})

    # home
    home = []
    for p in state.players:
        row = []
        for m in state.board.home[p]:
            if m is None:
                row.append(None)
            else:
                pi, mi = marble_index[m]
                row.append({"player": pi, "marble": mi})
        home.append(row)

    # hands
    hands = []
    for p in state.players:
        hands.append([card_to_dict(c) for c in p.hand])

    # teams list of tuples
    teams = []
    for p1, p2 in state.teams:
        teams.append((player_index[p1], player_index[p2]))

    switch_cards = []
    for p, c in state.switch_cards:
        switch_cards.append((player_index[p], card_to_dict(c)))

    winner = None
    if state.winner is not None:
        winner = (player_index[state.winner[0]], player_index[state.winner[1]])

    # deck + discard
    deck = [card_to_dict(c) for c in state.deck.cards]
    discard = [card_to_dict(c) for c in state.discard_pile]

    return {
        "players": players,
        "track": track,
        "home": home,
        "hands": hands,
        "deck": deck,
        "discard": discard,
        "draw_size": state.draw_size,
        "current_player": player_index[state.current_player],
        "last_started_player": (
            player_index[state.last_started_player]
            if state.last_started_player is not None
            else None
        ),
        "phase": state.phase.name,
        "num_rounds": state.num_rounds,
        "teams": teams,
        "switch_cards": switch_cards,
        "winner": winner,
    }


def state_from_dict(d: dict) -> GameState:
    # players + marbles
    players: list[Player] = []
    for pinfo in d["players"]:
        color = Colors[pinfo["color"]]
        marbles = [Marble(color) for _ in range(4)]
        players.append(Player(name=pinfo["name"], marbles=marbles))

    board = Board(players=players)

    # helper maps
    # (player_idx, marble_idx) -> marble object
    marble_obj = {}
    for pi, p in enumerate(players):
        for mi, m in enumerate(p.marbles):
            marble_obj[(pi, mi)] = m

    # track
    for i, cell in enumerate(d["track"]):
        if cell is None:
            board.track[i] = (None, None)
        else:
            p = players[cell["player"]]
            m = marble_obj[(cell["player"], cell["marble"])]
            board.track[i] = (m, p)

    # home
    for pi, row in enumerate(d["home"]):
        p = players[pi]
        for idx, cell in enumerate(row):
            if cell is None:
                board.home[p][idx] = None
            else:
                m = marble_obj[(cell["player"], cell["marble"])]
                board.home[p][idx] = m

    # hands
    for pi, cards in enumerate(d["hands"]):
        players[pi].hand = [card_from_dict(c) for c in cards]

    # deck + discard
    deck = Deck(cards=[card_from_dict(c) for c in d["deck"]])
    discard = [card_from_dict(c) for c in d["discard"]]

    # current / last / phase
    current_player = players[d["current_player"]]
    last_started_player = (
        players[d["last_started_player"]]
        if d["last_started_player"] is not None
        else None
    )
    phase = GamePhase[d["phase"]]
    teams = []
    for p1_idx, p2_idx in d["teams"]:
        teams.append((players[p1_idx], players[p2_idx]))

    switch_cards = []
    for p_idx, c in d["switch_cards"]:
        switch_cards.append((players[p_idx], card_from_dict(c)))

    winner = None
    if d["winner"] is not None:
        winner = (players[d["winner"][0]], players[d["winner"][1]])

    return GameState(
        players=players,
        board=board,
        deck=deck,
        discard_pile=discard,
        draw_size=d["draw_size"],
        current_player=current_player,
        last_started_player=last_started_player,
        phase=phase,
        num_rounds=d["num_rounds"],
        teams=teams,
        winner=winner,
    )


def save_state(state: GameState, path: str = "savegame.json") -> None:
    data = state_to_dict(state)
    Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_state(path: str = "savegame.json") -> GameState:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    state = state_from_dict(data)
    if state.phase in (GamePhase.TURN, GamePhase.PLAY):
        state.cp_actions = legal_actions(state)

    return state
