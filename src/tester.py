import chess
import chess.engine
import chess.pgn
import numpy as np
from datetime import datetime
import os

from lib import fen_to_input
from mcts.mcts import MCTS
from mcts.edge import Edge
from model.agent import Agent
import logging
from utils import config

class ChessEloTester:
    def __init__(self, model_path: str, stockfish_elo: int, pgn_target: str, num_rounds: int):
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        
        self.logger.info(f"Initializing ChessEloTester with model_path={model_path}, "
                        f"stockfish_elo={stockfish_elo}, pgn_target={pgn_target}, num_rounds={num_rounds}")
        
        self.logger.info("Loading pretrained model...")
        self.agent = Agent(model_path=model_path)
        
        # Initialize Stockfish
        self.stockfish_elo = stockfish_elo
        stockfish_path = os.getenv("STOCKFISH_PATH", "stockfish")  # Set STOCKFISH_PATH env var
        self.logger.info(f"Initializing Stockfish at ELO {stockfish_elo}...")
        try:
            self.stockfish = chess.engine.SimpleEngine.popen_uci(stockfish_path)
            self.stockfish.configure({"UCI_LimitStrength": True, "UCI_Elo": stockfish_elo})
        except Exception as e:
            self.logger.error(f"Failed to initialize Stockfish: {e}")
            raise
        
        self.pgn_target = pgn_target
        self.num_rounds = num_rounds
        self.results = []
        self.simulations_per_move = 100  # Fixed for ELO testing
        self.logger.info("Setup complete.")

    def predict(self, board: chess.Board) -> tuple[np.ndarray, float]:
        """
        Predict policy and value for a given board state using the model.
        
        Args:
            board (chess.Board): Current chess board.
        
        Returns:
            tuple: (policy, value) where policy is a numpy array and value is a scalar in [-1, 1].
        """
        self.logger.debug("Predicting for board state...")
        # Convert board to model input (assuming FEN-based input)
        fen = board.fen()
        input_state = fen_to_input(fen)  # Simplified; replace with your input conversion
        try:
            policy, value = self.model.predict(input_state)
            self.logger.debug(f"Prediction: policy shape={policy.shape}, value={value}")
            return policy, float(value)
        except Exception as e:
            self.logger.error(f"Prediction failed: {e}")
            raise

    def play_game(self, opponent: str = "stockfish", as_white: bool = True) -> str:
        """
        Play a single game against Stockfish or a human.
        
        Args:
            opponent (str): 'stockfish' or 'human'.
            as_white (bool): True if model plays as white, False as black.
        
        Returns:
            str: Game result ('1-0', '0-1', '1/2-1/2').
        """
        self.logger.info(f"Starting game: Model as {'White' if as_white else 'Black'} vs {opponent}")
        board = chess.Board()
        game = chess.pgn.Game()
        game.headers["Event"] = f"Model vs {opponent} ELO {self.stockfish_elo}"
        game.headers["White"] = "Model" if as_white else "Stockfish"
        game.headers["Black"] = "Stockfish" if as_white else "Model"
        game.headers["Date"] = datetime.now().strftime("%Y.%m.%d")
        
        mcts = MCTS(agent=self, state=board.fen(), stochastic=False)
        node = game
        previous_moves = (None, None)

        while not board.is_game_over():
            if (as_white and board.turn == chess.WHITE) or (not as_white and board.turn == chess.BLACK):
                # Model's turn
                self.logger.info(f"Model's turn at move {board.fullmove_number}...")
                mcts.simulate(self.simulations_per_move)
                moves = mcts.root.edges
                if not moves:
                    self.logger.warning("No moves available; ending game")
                    break
                
                sum_visits = sum(e.N for e in moves)
                probs = [e.N / sum_visits for e in moves]
                best_move = moves[np.argmax(probs)]  # Deterministic for testing
                move = best_move.action
                self.logger.info(f"Model plays {move.uci()}")
                
                # Update MCTS root
                try:
                    mcts.root = best_move.output_node
                except AttributeError:
                    self.logger.warning("Move not in tree; creating new MCTS")
                    mcts = MCTS(agent=self, state=board.fen(), stochastic=False)
                
                board.push(move)
                node = node.variation(move)
                previous_moves = (previous_moves[1], best_move)
            else:
                # Opponent's turn
                if opponent == "stockfish":
                    self.logger.info(f"Stockfish's turn at move {board.fullmove_number}...")
                    result = self.stockfish.play(board, chess.engine.Limit(time=0.1))
                    move = result.move
                    self.logger.info(f"Stockfish plays {move.uci()}")
                else:
                    # Human input (optional)
                    self.logger.info("Enter human move (UCI format, e.g., 'e2e4'): ")
                    uci = input().strip()
                    try:
                        move = chess.Move.from_uci(uci)
                        if move not in board.legal_moves:
                            self.logger.error("Invalid move; ending game")
                            break
                    except ValueError:
                        self.logger.error("Invalid UCI format; ending game")
                        break
                
                board.push(move)
                node = node.variation(move)
                # Update MCTS root
                try:
                    mcts.root = mcts.root.get_edge(move).output_node
                except AttributeError:
                    self.logger.warning("Opponent move not in tree; creating new MCTS")
                    mcts = MCTS(agent=self, state=board.fen(), stochastic=False)
                previous_moves = (previous_moves[1], Edge(action=move, input_node=None, output_node=None))

        result = board.result()
        game.headers["Result"] = result
        self.logger.info(f"Game ended with result {result}")
        self.save_game_to_pgn(game)
        return result

    def save_game_to_pgn(self, game: chess.pgn.Game):
        """
        Save a game to the PGN file.
        
        Args:
            game (chess.pgn.Game): Game to save.
        """
        self.logger.info(f"Saving game to {self.pgn_target}...")
        try:
            with open(self.pgn_target, "a") as f:
                print(game, file=f, end="\n\n")
        except Exception as e:
            self.logger.error(f"Failed to save PGN: {e}")
            raise

    def run_tests(self):
        """
        Run the specified number of rounds, alternating colors, and print a summary.
        """
        self.logger.info(f"Starting {self.num_rounds} rounds against Stockfish ELO {self.stockfish_elo}")
        for i in range(self.num_rounds):
            as_white = (i % 2 == 0)  # Alternate colors
            self.logger.info(f"Round {i+1}/{self.num_rounds}: Model as {'White' if as_white else 'Black'}")
            result = self.play_game(opponent="stockfish", as_white=as_white)
            self.results.append((result, as_white))
        
        # Print summary
        self.logger.info("=== Test Summary ===")
        scores = {"1-0": 1, "1/2-1/2": 0.5, "0-1": 0}
        total_score = 0
        white_games = 0
        black_games = 0
        white_score = 0
        black_score = 0

        for result, as_white in self.results:
            score = scores[result] if as_white else 1 - scores[result]
            total_score += score
            if as_white:
                white_games += 1
                white_score += score
            else:
                black_games += 1
                black_score += score

        avg_score = total_score / self.num_rounds if self.num_rounds > 0 else 0
        white_avg = white_score / white_games if white_games > 0 else 0
        black_avg = black_score / black_games if black_games > 0 else 0
        
        # Estimate ELO (simplified)
        expected_score = avg_score
        elo_diff = -400 * np.log10(1 / expected_score - 1) if 0 < expected_score < 1 else 0
        estimated_elo = self.stockfish_elo + elo_diff

        self.logger.info(f"Total Games: {self.num_rounds}")
        self.logger.info(f"Model as White: {white_games} games, Average Score: {white_avg:.2f}")
        self.logger.info(f"Model as Black: {black_games} games, Average Score: {black_avg:.2f}")
        self.logger.info(f"Overall Average Score: {avg_score:.2f}")
        self.logger.info(f"Estimated ELO: {estimated_elo:.0f} (vs Stockfish ELO {self.stockfish_elo})")
        self.logger.info(f"PGN saved to: {self.pgn_target}")
        self.logger.info("Test complete.")

    def cleanup(self):
        """
        Close Stockfish engine.
        """
        self.logger.info("Cleaning up...")
        try:
            self.stockfish.quit()
        except Exception as e:
            self.logger.error(f"Failed to close Stockfish: {e}")

    def play_human(self):
        """
        Play a single game against a human (optional).
        """
        self.logger.info("Starting human game...")
        result = self.play_game(opponent="human", as_white=True)
        self.logger.info(f"Human game ended with result {result}")


if __name__ == "__main__":
    tester = ChessEloTester(
        model_path="./checkpoints/model.pth",
        stockfish_elo=1350,
        pgn_target="games.pgn",
        num_rounds=10
    )
    try:
        tester.run_tests()
    finally:
        tester.cleanup()