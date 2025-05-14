import os
from chess.pgn import Game as ChessGame
import uuid
import numpy as np
import functools
import time

from environment.env import Environment
from mcts.edge import Edge
from mcts.mcts import MCTS
from model.agent import Agent
from utils import config

from model.train import ChessDataset, Trainer

# PyTorch-compatible time_function decorator (replacing lib.time_function if needed)
def time_function(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        print(f"{func.__name__} took {time.time() - start_time:.3f} seconds")
        return result
    return wrapper

class Game:
    def __init__(self, env: Environment, white: Agent, black: Agent):
        self.env = env
        self.white = white
        self.black = black
        self.memory = []
        self.reset()

    def reset(self):
        self.env.reset()
        self.turn = self.env.board.turn

    @staticmethod
    def get_winner(result: str) -> int:
        return 1 if result == "1-0" else -1 if result == "0-1" else 0

    @time_function
    def play_one_game(self, stochastic: bool = True, train: bool = False) -> int:
        self.reset()
        self.memory.append([])
        print(f'\n{self.env.board}')
        counter, previous_edges, full_game = 0, (None, None), True
        while not self.env.board.is_game_over():
            previous_edges = self.play_move(stochastic=stochastic, previous_moves=previous_edges)
            print(f'\n{self.env.board}')
            print(f'Value according to white: {self.white.mcts.root.value}')
            print(f'Value according to black: {self.black.mcts.root.value}')

            counter += 1
            if counter > config.MAX_GAME_MOVES:
                winner = Environment.estimate_winner(self.env.board)
                print(f'Game over by move limit ({config.MAX_GAME_MOVES}). Result: {winner}')
                full_game = False
                break

        winner = -1
        if full_game:
            winner = Game.get_winner(self.env.board.result())
            print(f'Game over. Result: {winner}')

        for index, element in enumerate(self.memory[-1]):
            self.memory[-1][index] = (element[0], element[1], winner)

        game = ChessGame()
        game.setup(self.env.fen)
        if self.env.board.move_stack:  # Check if there are moves to add
            node = game.add_variation(self.env.board.move_stack[0])
            for move in self.env.board.move_stack[1:]:
                node = node.add_variation(move)

        print(f'INFO: {game}')
        self.save_game(name="game", full_game=full_game)
        return winner

    def play_move(self, stochastic: bool = True, previous_moves=(None, None), save_moves=True):
        current_player = self.white if self.turn else self.black

        if previous_moves[0] is None or previous_moves[1] is None:
            current_player.mcts = MCTS(current_player, state=self.env.board.fen(), stochastic=stochastic)
        else:
            try:
                node = current_player.mcts.root.get_edge(previous_moves[0].action).output_node
                node = node.get_edge(previous_moves[1].action).output_node
                current_player.mcts.root = node  # Fixed typo: output_node -> node
            except AttributeError:
                print(f'WARN: Node does not exist in tree, continuing with a new tree')
                current_player.mcts = MCTS(current_player, state=self.env.board.fen(), stochastic=stochastic)
        current_player.run_simulations(config.SIMULATIONS_PER_MOVE)
        moves = current_player.mcts.root.edges

        if save_moves:
            self.save_to_memory(self.env.board.fen(), moves)

        sum_move_visits = sum(e.N for e in moves)
        probs = [e.N / sum_move_visits for e in moves]

        if stochastic:
            best_move = np.random.choice(moves, p=probs)
        else:
            best_move = moves[np.argmax(probs)]

        print(f"{'White' if self.turn else 'Black'} played {self.env.board.fullmove_number}. {best_move.action}")
        self.env.step(best_move.action)
        self.turn = not self.turn

        return (previous_moves[1], best_move)

    def save_to_memory(self, state, moves) -> None:
        sum_move_visits = sum(e.N for e in moves)
        search_probabilities = {e.action.uci(): e.N / sum_move_visits for e in moves}
        self.memory[-1].append((state, search_probabilities, None))

    def save_game(self, name: str = "game", full_game: bool = False) -> None:
        """
        Save the internal memory to a .npy file.
        """
        game_id = f"{name}-{str(uuid.uuid4())[:8]}"
        if full_game:
            with open("full_games.txt", "a") as f:
                f.write(f"{game_id}.npy\n")
        np.save(os.path.join(config.MEM_DIR, game_id), self.memory[-1])
        print(f"Game saved to {os.path.join(config.MEM_DIR, game_id)}.npy")
        print(f"Memory size: {len(self.memory)}")
