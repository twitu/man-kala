import numpy as np
import random
import logging

from typing import Tuple
from copy import deepcopy

TILE_START_COUNT = 3
PLAYER_COUNT = 2
PER_PLAYER_TILE = 6
PLAYER_HOME_INDEX = PER_PLAYER_TILE // 2
BOARD_LENGTH = PER_PLAYER_TILE * PLAYER_COUNT
FIRST_PLAYER_AGENT = True
RANDOM_SEED = 23

type OBSERVATION = tuple[
    int, int, int, int, int, int, int, int, int, int, int, int, int, int
]
OBSERVATION_LEN = 14


class GameBoard:
    def __init__(self) -> None:
        self.board = np.array([TILE_START_COUNT] * BOARD_LENGTH)

    # Game is over if there are no more beads on the board
    def over(self) -> bool:
        return np.all(self.board == 0)

    # Player `index` plays a round starting at tile index `start_tile_index` relative to them
    # Returns the points earned by the player
    def play_turn(self, player_index, start_tile_index) -> int:
        logging.debug(f"Playing {start_tile_index} for Player {player_index}")
        if 0 <= start_tile_index <= PER_PLAYER_TILE:
            offset = player_index * PER_PLAYER_TILE
            start_tile_index += offset
            player_home_index = PLAYER_HOME_INDEX + offset
            player_score = 0
            logging.debug(
                f"Playing absolute {start_tile_index} for Player {player_index}"
            )
            logging.debug(f"{self}")

            while self.board[start_tile_index] != 0:
                beads = self.board[start_tile_index]
                self.board[start_tile_index] = 0
                cur_index = start_tile_index
                logging.debug(
                    f"beads: {beads}, cur_index: {cur_index}, score: {player_score}"
                )
                logging.debug(f"{self}")

                while beads > 0:
                    cur_index = (cur_index + 1) % BOARD_LENGTH
                    # special case for home index
                    # if one bead increment score and stop
                    # if two or more beads increment score and continue
                    if cur_index == player_home_index and cur_index != start_tile_index:
                        player_score += 1
                        beads -= 1
                        if beads == 0:
                            logging.debug(f"Final score: {player_score}")
                            logging.debug(f"{self}")
                            return player_score

                    # add bead to tile
                    if beads > 0:
                        self.board[cur_index] += 1
                        beads -= 1

                    logging.debug(
                        f"beads: {beads}, cur_index: {cur_index}, score: {player_score}"
                    )
                    logging.debug(f"{self}")

                # if no beads in hand and current tile has more than 1 bead
                # pick up the beads and continue cycle
                if self.board[cur_index] > 1:
                    start_tile_index = cur_index
                else:
                    break

            logging.debug(f"Final score: {player_score}")
            logging.debug(f"{self}")
            return player_score
        else:
            raise Exception(f"Player {player_index} cannot start at {start_tile_index}")

    def valid_moves(self, player_index) -> list[int]:
        offset = player_index * PER_PLAYER_TILE
        return [
            tile - offset
            for tile in range(offset, offset + PER_PLAYER_TILE)
            if self.board[tile] > 0
        ]

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
    def __init__(
        self,
        players,
    ) -> None:
        assert len(players) == PLAYER_COUNT
        self.players = players
        self.board = GameBoard()
        self.score = [0] * len(players)
        self.turn = random.randint(0, PLAYER_COUNT - 1)
        self.tiles_played = [[] for _ in range(PLAYER_COUNT)]

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

    def simulate_game(self):
        while not self.board.over():
            self.play_next_turn()

        for i in range(PLAYER_COUNT):
            logging.info(
                f"Player {self.players[i].name} played: {self.tiles_played[i]}"
            )

    def play_next_turn(self):
        player = self.players[self.turn]
        play_tile_index = player.play_turn(self)

        logging.info(self)
        logging.info(f"{self.players[self.turn].name} playing {play_tile_index}\n")

        result = self.play_tile(play_tile_index)
        logging.info(self)
        return result

    # Play given tile index for current turn
    # Return player's index and new score
    def play_tile(self, play_tile_index: int) -> Tuple[int, int]:
        self.tiles_played[self.turn].append(play_tile_index)
        score = self.board.play_turn(self.turn, play_tile_index)
        self.score[self.turn] += score
        result = (self.turn, self.score[self.turn])
        self.turn = (self.turn + 1) % PLAYER_COUNT
        return result

    # Reset environment for open ai gym
    def reset(self, *args, **kwargs):
        self.board = GameBoard()
        self.score = [0] * len(self.players)
        self.turn = random.randint(0, PLAYER_COUNT - 1)
        self.tiles_played = [[] for _ in range(PLAYER_COUNT)]

        return self.observation, {}

    # Record observation for open ai gym
    # Returns the player scores and board state
    @property
    def observation(self) -> OBSERVATION:
        obs = []
        first_player_tiles = deepcopy(self.board.board[0:PER_PLAYER_TILE])
        second_player_tiles = deepcopy(self.board.board[PER_PLAYER_TILE:])
        if FIRST_PLAYER_AGENT:
            obs.append(self.score[0])
            obs.append(self.score[1])
            obs.extend(first_player_tiles)
            obs.extend(second_player_tiles)
        else:
            obs.append(self.score[1])
            obs.append(self.score[0])
            obs.extend(second_player_tiles)
            obs.extend(first_player_tiles)
        return tuple(obs)

    @property
    def action_space(self):
        agent_index = 0 if FIRST_PLAYER_AGENT else 1
        return self.board.valid_moves(agent_index)

    def step(self, action: int):
        self.tiles_played[self.turn].append(action)
        score = self.board.play_turn(self.turn, action)
        self.score[self.turn] += score
        result = (self.turn, self.score[self.turn])
        self.turn = (self.turn + 1) % PLAYER_COUNT
        return self.observation, score, self.board.over(), False, {}


class ReplayPlayer:
    def __init__(self, played_tiles: list[int]):
        self.played_tiles = played_tiles
        self.current_turn = 0
        self.name = "ReplayPlayer"

    def play_turn(self, _sim: GameSimulator) -> int:
        play_tile = self.played_tiles[self.current_turn]
        self.current_turn += 1
        return play_tile


class InteractivePlayer:
    def __init__(self) -> None:
        self.full_name = "InteractivePlayer"
        self.name = "Intrct"

    def play_turn(self, _sim: GameSimulator) -> int:
        while True:
            tile = input(f"Enter a value between 0 and {PER_PLAYER_TILE}: ")
            try:
                return int(tile)
            except Exception as error:
                logging.error(f"Error {error}")


class RandomPlayer:
    def __init__(self, rand_seed) -> None:
        self.full_name = f"RandomPlayer({rand_seed})"
        self.name = f"R({rand_seed})"
        self.gen = random.Random(rand_seed)

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
            return self.gen.choice(choices)
        else:
            return 0


class LocalMaximaPlayer:
    def __init__(self) -> None:
        self.full_name = "LocalMaximaPlayer"
        self.name = f"LoMX"

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


class MinimaxPlayer:
    def __init__(self, max_depth=5) -> None:
        # minimax is designed for only two players
        assert PLAYER_COUNT == 2
        self.full_name = f"MinimaxPlayer({max_depth})"
        self.name = f"MnMX({max_depth})"
        self.max_depth = max_depth

    def play_turn(self, sim: GameSimulator) -> int:
        my_index = sim.turn
        opp_index = (sim.turn + 1) % 2

        # Minimax for a given board state and depth returns the maximum points
        # by which I am leading. If depth is even it is my turn and I want to
        # maximize my lead. If depth is odd then it is the opponent's turn
        # and they want to minimize my lead.
        def minimax(
            board: GameBoard,
            score_board: Tuple[int, int],
            depth: int,
            min_score: int,
            max_score: int,
        ) -> int:
            # If game reaches end state
            # Return points lead
            if board.over() or depth >= self.max_depth:
                (me, opp) = score_board
                return me - opp
            else:
                (me, opp) = score_board
                my_turn = depth % 2 == 0
                # My turn play a tile and add to my score
                if my_turn:
                    valid_tiles = board.valid_moves(my_index)
                    score = min_score

                    # if no valid tiles skip to opponents move
                    if not valid_tiles:
                        return minimax(
                            board, score_board, depth + 1, min_score, max_score
                        )

                    for tile in valid_tiles:
                        board_copy = deepcopy(board)
                        new_score_board = (
                            me + board_copy.play_turn(my_index, tile),
                            opp,
                        )
                        new_score = minimax(
                            board_copy, new_score_board, depth + 1, score, max_score
                        )

                        if new_score > max_score:
                            return max_score
                        elif new_score > score:
                            score = new_score

                    return score
                else:
                    valid_tiles = board.valid_moves(opp_index)
                    score = max_score

                    # if no valid tiles skip to opponents move
                    if not valid_tiles:
                        return minimax(
                            board, score_board, depth + 1, min_score, max_score
                        )

                    for tile in valid_tiles:
                        board_copy = deepcopy(board)
                        new_score_board = (
                            me,
                            opp + board_copy.play_turn(opp_index, tile),
                        )
                        new_score = minimax(
                            board_copy, new_score_board, depth + 1, min_score, score
                        )

                        if new_score < min_score:
                            return min_score
                        elif new_score < score:
                            score = new_score

                    return score

        valid_tiles = sim.board.valid_moves(my_index)
        scores = []
        max_score = PER_PLAYER_TILE * TILE_START_COUNT * PLAYER_COUNT
        for tile in valid_tiles:
            board_copy = deepcopy(sim.board)
            score = board_copy.play_turn(my_index, tile)
            scores.append(
                (minimax(board_copy, (score, 0), 1, -max_score, max_score), tile)
            )

        if scores:
            (_, play_tile) = max(scores)
            return play_tile
        else:
            return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p1 = InteractivePlayer()
    p2 = MinimaxPlayer(7)
    sim = GameSimulator([p1, p2])
    logging.info(
        f"{PLAYER_COUNT} players with {PER_PLAYER_TILE} tiles each and {TILE_START_COUNT} starting beads per tile\n"
    )
    sim.simulate_game()
