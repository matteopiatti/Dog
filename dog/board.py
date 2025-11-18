from dog.objects import Marble, Player


class Board:
    NUM_FIELDS = 64
    START_FIELD_NUMBERS = {
        0: 0,
        1: 16,
        2: 32,
        3: 48,
    }

    def __init__(self, players: list["Player"]):
        self.track = [(None, None)] * self.NUM_FIELDS
        self.home = {player: [None] * 4 for player in players}
        self.start_fields: dict[Player, int] = {
            player: self.START_FIELD_NUMBERS[idx] for idx, player in enumerate(players)
        }
        self.blocked_fields = set()

    def pos_of_marble(self, marble: Marble) -> int | None:
        for i, (m, p) in enumerate(self.track):
            if m is marble:
                return i
        return None

    def pos_of_home_marble(self, marble: Marble, player: "Player") -> int | None:
        for i, m in enumerate(self.home[player]):
            if m is marble:
                return i
        return None

    def pos_empty(self, pos: int) -> bool:
        m, _ = self.track[pos]
        return m is None

    def marble_can_move_home(self, marble: Marble, player: "Player", steps) -> bool:
        startfield = self.start_fields[player]
        if startfield in self.blocked_fields:
            return False
        pos = self.pos_of_marble(marble)
        if pos is None:
            return False
        distance_to_start = (startfield - pos) % self.NUM_FIELDS
        distance_to_first_marble = next(
            (i for i, v in enumerate(self.home[player]) if v is not None), 4
        )
        return (
            steps <= distance_to_start + distance_to_first_marble
            and steps > distance_to_start
        )

    def get_free_player_marble(self, player: "Player") -> Marble | None:
        in_play = self.player_marbles_in_play(player)
        in_home = self.home[player]
        for marble in player.marbles:
            if marble not in in_play and marble not in in_home:
                return marble
        return None

    def player_marbles_in_play(self, player: "Player"):
        return [m for (m, p) in self.track if p is player and m is not None]

    def player_has_startable_marble(self, player: "Player") -> bool:
        if self.start_fields[player] in self.blocked_fields:
            return False
        return self.get_free_player_marble(player) is not None

    def player_finished_marbles(self, player: "Player") -> int:
        home = self.home[player]
        n = len(home)
        finished = 0

        for i in range(n):
            if home[i] is not None:
                if all(slot is not None for slot in home[i + 1 :]):
                    finished += 1

        return finished

    def team_finished_marbles(self, team: tuple["Player", "Player"]) -> int:
        return self.player_finished_marbles(team[0]) + self.player_finished_marbles(
            team[1]
        )

    def total_distance_to_home(self, player: "Player") -> int:
        startfield = self.start_fields[player]
        total = 0
        for m in self.player_marbles_in_play(player):
            pos = self.pos_of_marble(m)
            if pos is None:
                continue
            distance = (startfield - pos) % self.NUM_FIELDS
            total += distance
        return total
