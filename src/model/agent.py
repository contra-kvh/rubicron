import torch
import chess
import numpy as np

from mcts.mcts import MCTS
from model.builder import RLModel
from utils import config
from model.local_pred import predict_local

class Agent:
    def __init__(self, model_path: str | None = None, state=chess.STARTING_FEN):
        self.model: RLModel = RLModel(config.INPUT_SHAPE, config.OUTPUT_SHAPE).to(config.DEVICE)
        if model_path is not None:
            self.model.load_state_dict(torch.load(model_path, map_location=config.DEVICE))
        self.local_prediction = True
        self.mcts = MCTS(self, state=state)

    def run_simulations(self, n: int = 1):
        print(f'Running {n} simulations...')
        self.mcts.simulate(n)

    def predict(self, data):
        assert self.model is not None, "Model is not loaded"
        p, v = predict_local(self.model, data)
        return p.cpu().numpy(), v[0][0]