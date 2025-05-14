import chess
import numpy as np
from tqdm import tqdm
from graphviz import Digraph
import threading

from mcts.mapper import Mapping
from utils import config
from environment.env import Environment
from mcts.node import Node
from mcts.edge import Edge



class MCTS:
    def __init__(self, agent: "Agent", state: str = chess.STARTING_FEN, stochastic: bool = False):
        self.root = Node(state = state)
        self.game_path: list[Edge] = []
        self.cur_board: chess.Board | None = None
        self.agent = agent
        self.stochastic = stochastic
        self.outputs = []


    def simulate(self, n: int) -> None:
        for _ in tqdm(range(n)):
            self.game_path = []
            leaf = self.select_child(self.root)
            leaf.N += 1
            leaf = self.expand(leaf)
            leaf = self.backpropagate(leaf, leaf.value)


    def select_child(self, node: Node) -> Node:
        while not node.is_leaf():
            if not len(node.edges):
                return node
            
            noise = [1 for _ in range(len(node.edges))]
            if self.stochastic and node == self.root:
                noise = np.random.dirichlet([config.DIRICHLET_NOISE] * len(node.edges))

            best_edge = None
            best_score = -np.inf

            for i, edge in enumerate(node.edges):
                if edge.upper_confidence_bound(noise[i]) > best_score:
                    best_score = edge.upper_confidence_bound(noise[i])
                    best_edge = edge

            if best_edge is None:
                raise Exception("No best edge found")

            node = best_edge.output_node
            self.game_path.append(best_edge)
        return node

    
    def map_valid_move(self, move: chess.Move) -> None:
        assert self.cur_board is not None, "Current board is None"

        # print('filtering valid moves...', end='\r')
        from_sq = move.from_square
        to_sq = move.to_square

        plane_idx: int = -1
        piece = self.cur_board.piece_at(from_sq)
        direction = None

        if piece is None:
            raise Exception(f'No piece at {from_sq}')

        if move.promotion and move.promotion != chess.QUEEN:
            piece_type, direction = Mapping.get_underpromotion_move(move.promotion, from_sq, to_sq)
            plane_idx = Mapping.mapper[piece_type][1 - direction]
        else:
            if piece.piece_type == chess.KNIGHT:
                direction = Mapping.get_knight_move(from_sq, to_sq)
                plane_idx = Mapping.mapper[direction]
            else:
                direction, distance = Mapping.get_queenlike_move(from_sq, to_sq)
                plane_idx = Mapping.mapper[direction][np.abs(distance) - 1]

        row = from_sq % 8
        col = 7 - (from_sq // 8)
        self.outputs.append((move, plane_idx, row, col))


    def probabilities_to_actions(self, probabilities: np.ndarray, board: str) -> dict:
        probabilities = probabilities.reshape(config.n_output_planes, config.n, config.n)
        actions = {}
        self.cur_board = chess.Board(board)
        valid_moves = self.cur_board.generate_legal_moves()
        self.outputs = []
        threads = []
        while True:
            try:
                move = next(valid_moves)
            except StopIteration:
                break
            thread = threading.Thread(target=self.map_valid_move, args=(move,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        for move, plane_idx, col, row in self.outputs:
            actions[move.uci()] = probabilities[plane_idx, row, col]

        return actions


    def expand(self, leaf: Node) -> Node:
        # print("Expanding node...", end='\r')
        board = chess.Board(leaf.state)
        possible_actions = list(board.generate_legal_moves())

        if not len(possible_actions):
            assert board.is_game_over(), "Game is not over but there are no possible moves"
            outcome = board.outcome(claim_draw=True)
            if outcome is None:
                leaf.value = 0
            else:
                leaf.value = 1 if outcome.winner == chess.WHITE else -1
            return leaf

        input_state = Environment.fen_to_input(leaf.state)
        assert self.agent is not None, "Agent is None"
        p, v = self.agent.predict(input_state)

        actions = self.probabilities_to_actions(p, leaf.state)
        # print(f'Model predictions: {p}', end='\r')
        # print(f'Value of state: {v}', end='\r')
        leaf.value = v

        for action in possible_actions:
            new_state = leaf.step(action)
            leaf.add_child(Node(new_state), action, actions[action.uci()])
        return leaf


    def backpropagate(self, leaf: Node, value: float) -> Node:
        # print("Backpropagating...", end='\r')
        for edge in self.game_path:
            edge.input_node.N += 1
            edge.N += 1
            edge.W += value
        return leaf

    def plot_node(self, dot: Digraph, node: Node):
        dot.node(f"{node.state}", f"N")
        for edge in node.edges:
            dot.edge(str(edge.input_node.state), str(
                edge.output_node.state), label=edge.action.uci())
            dot = self.plot_node(dot, edge.output_node)
        return dot

    def plot_tree(self, save_path: str = "tests/mcts_tree.gv") -> None:
        # print("Plotting tree...")
        # tree plotting
        dot = Digraph(comment='Chess MCTS Tree')
        print(f"# of nodes in tree: {len(self.root.get_all_children())}")

        # recursively plot the tree
        dot = self.plot_node(dot, self.root)
        dot.save(save_path)
