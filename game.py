from typing import Tuple
import numpy as np
import random
from copy import deepcopy

TILE_START_COUNT = 3
PLAYER_COUNT = 2
PER_PLAYER_TILE = 6
PLAYER_HOME_INDEX = PER_PLAYER_TILE // 2
BOARD_LENGTH = PER_PLAYER_TILE * PLAYER_COUNT


class GameBoard:
    def __init__(self) -> None:
        self.board = np.array([TILE_START_COUNT] * BOARD_LENGTH)

    # Game is over if there are no more beads on the board
    def over(self) -> bool:
        return np.all(self.board == 0)

    # Player `index` plays a round starting at tile index `start_tile_index` relative to them
    def play_turn(self, player_index, start_tile_index) -> int:
        if 0 <= start_tile_index <= PER_PLAYER_TILE:
            offset = player_index * PER_PLAYER_TILE
            start_tile_index += offset
            player_home_index = PLAYER_HOME_INDEX + offset
            player_score = 0

            while self.board[start_tile_index] != 0:
                beads = self.board[start_tile_index]
                self.board[start_tile_index] = 0
                cur_index = start_tile_index

                while beads > 0:
                    cur_index = (cur_index + 1) % BOARD_LENGTH
                    # special case for home index
                    # if one bead increment score and stop
                    # if two or more beads increment score and continue
                    if cur_index == player_home_index and cur_index != start_tile_index:
                        player_score += 1
                        beads -= 1

                    # add bead to tile
                    if beads > 0:
                        self.board[cur_index] += 1
                        beads -= 1

                # if no beads in hand and current tile has more than 1 bead
                # pick up the beads and continue cycle
                if self.board[cur_index] > 1:
                    start_tile_index = cur_index
                else:
                    break

            return player_score
        else:
            raise Exception(f"Player {player_index} cannot start at {start_tile_index}")

    def __str__(self) -> str:
        str_builder = ""

        for i in range(PLAYER_COUNT):
            str_builder += f"Player {i} tiles: "
            start_index = i * PER_PLAYER_TILE
            str_builder += ",".join(
                [
                    str(val)
                    for val in self.board[start_index : start_index + PER_PLAYER_TILE]
                ]
            )
            str_builder += "\n"

        return str_builder


class GameSimulator:
    def __init__(self, players) -> None:
        assert len(players) == PLAYER_COUNT
        self.players = players
        self.board = GameBoard()
        self.score = [0] * len(players)
        self.turn = 0

    def __str__(self) -> str:
        str_builder = ""
        str_builder += "\n".join(
            [
                f"({i}) {player.name} has {score}"
                for (i, (player, score)) in enumerate(zip(self.players, self.score))
            ]
        )
        str_builder += f"\n{self.board}"
        return str_builder

    def play(self):
        while not self.board.over():
            player = self.players[self.turn]
            play_tile_index = player.play_turn(self)

            print(self)
            print(f"{self.players[self.turn].name} playing {play_tile_index}\n")

            score = self.board.play_turn(self.turn, play_tile_index)
            self.score[self.turn] += score

            self.turn = (self.turn + 1) % PLAYER_COUNT

        print(self)

class RandomPlayer:
    def __init__(self, rand_seed) -> None:
        self.name = "RandomPlayer"
        random.seed(rand_seed)

    def play_turn(self, sim: GameSimulator) -> int:
        index = sim.turn * PER_PLAYER_TILE
        choices = [
            i
            for (i,), beads in np.ndenumerate(
                sim.board.board[index : index + PER_PLAYER_TILE]
            )
            if beads > 0
        ]
        if choices:
            return random.choice(choices)
        else:
            return 0


class LocalMaximaPlayer:
    def __init__(self) -> None:
        self.name = "LocalMaximaPlayer"

    def play_turn(self, sim: GameSimulator) -> int:
        scores = []
        offset = sim.turn * PER_PLAYER_TILE
        for play_tile in range(PER_PLAYER_TILE):
            if sim.board.board[offset + play_tile] > 0:
                board_copy = deepcopy(sim.board)
                score = board_copy.play_turn(sim.turn, play_tile)
                scores.append((score, play_tile))

        if scores:
            (_, play_tile) = max(scores)
            return play_tile
        else:
            return 0


if __name__ == "__main__":
    p1 = RandomPlayer(23)
    p2 = LocalMaximaPlayer()
    sim = GameSimulator([p1, p2])
    print(
        f"{PLAYER_COUNT} players with {PER_PLAYER_TILE} tiles and {TILE_START_COUNT} starting beads per tile\n"
    )
    sim.play()
