import chess
from chess import Move
from mcts.edge import Edge

class Node:
    def __init__(self, state: str):
        self.state = state
        self.turn = chess.Board(state).turn
        self.edges: list[Edge] = []
        self.N: int = 0
        self.value: float = 0


    def step(self, action: Move) -> str:
        board = chess.Board(self.state)
        board.push(action)
        return board.fen()


    def if_game_over(self) -> bool:
        return chess.Board(self.state).is_game_over()


    def is_leaf(self) -> bool:
        return self.N == 0


    def add_child(self, child, action: Move, prior: float) -> Edge:
        edge = Edge(input_node=self, output_node=child, action=action, prior=prior)
        self.edges.append(edge)
        return edge


    def get_all_children(self):
        children = []
        for edge in self.edges:
            children.append(edge.output_node)
            children.append(edge.output_node.get_all_children())
        return children


    def get_edge(self, action: Move) -> Edge | None:
        for edge in self.edges:
            if edge.action == action:
                return edge
        return None

    def __eq__(self, other) -> bool:
        if not isinstance(other, Node):
            return False
        return self.state == other.state and self.turn == other.turn and self.edges == other.edges


