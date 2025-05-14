import os
import socket
import time
import numpy as np
import chess
import torch

from environment.game import Game
from environment.env import Environment as ChessEnv

from model.agent import Agent
from utils import config
from model.builder import RLModel


def setup(starting_position: str = chess.STARTING_FEN, local_predictions: bool = False) -> Game:
    """
    Setup function to set up a game.
    This can be used in both the self-play and puzzle-solving function.
    """
    # Set different random seeds for each process
    number = int.from_bytes(socket.gethostname().encode(), 'little')
    number *= os.getpid() if os.getpid() != 0 else 1
    number *= int(time.time())
    number %= 123456789
    
    np.random.seed(number)
    torch.manual_seed(number)
    print(f"========== > Setup. Test Random number: {np.random.randint(0, 123456789)}")

    # Create environment and game
    env = ChessEnv(fen=starting_position)

    model_path = os.path.join(config.MODEL_DIR, "model.pt")
    print(f"Loading model from {model_path}")
    white = Agent(local_predictions, model_path, env.board.fen())
    black = Agent(local_predictions, model_path, env.board.fen())

    return Game(env=env, white=white, black=black)

def self_play(local_predictions: bool = False, train: bool = False):
    """
    Continuously play games against itself.
    """
    game = setup(local_predictions=local_predictions)
    while True:
        game.play_one_game(stochastic=True, train=train)
