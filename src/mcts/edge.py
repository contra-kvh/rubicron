import chess
from chess import Move
import math

from utils import config


class Edge:
    def __init__(self, input_node: "Node", output_node: "Node", action: Move, prior: float):
        self.input_node = input_node
        self.output_node = output_node
        self.action = action
        self.P: float = prior
        self.N: int = 0
        self.W: float = 0.0
        self.player_turn = self.input_node.state.split(" ")[1] == 'w'


    def upper_confidence_bound(self, noise: float = 0.0) -> float:
        exploration_rate = math.log((1 + self.input_node.N + config.C_BASE) / config.C_BASE) + config.C_INIT
        ucb = exploration_rate * (self.P * noise) * (math.sqrt(self.input_node.N) / (1 + self.N))
        if self.player_turn == chess.WHITE:
            return self.W / (self.N + 1) + ucb
        else:
            return -self.W / (self.N + 1) + ucb


    def __eq__(self, other) -> bool:
        if not isinstance(other, Edge):
            return False
        return self.action == other.action and self.input_node == other.input_node and self.output_node == other.output_node

    def __str__(self):
        return f"{self.action.uci()}: Q={self.W / self.N if self.N != 0 else 0}, N={self.N}, W={self.W}, P={self.P}, U = {self.upper_confidence_bound()}"

    def __repr__(self):
        return f"{self.action.uci()}: Q={self.W / self.N if self.N != 0 else 0}, N={self.N}, W={self.W}, P={self.P}, U = {self.upper_confidence_bound()}"

