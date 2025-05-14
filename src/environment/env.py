import chess
from chess import Move
import numpy as np

from utils import config


class Environment:
    def __init__(self, fen: str = chess.STARTING_FEN):
        self.fen = fen
        self.reset()


    def reset(self):
        self.board = chess.Board(self.fen)


    def step(self, action: Move) -> chess.Board:
        self.board.push(action)
        return self.board


    @staticmethod
    def fen_to_input(fen: str) -> np.ndarray:
        board = chess.Board(fen)

        turn = np.full((8, 8), True) if board.turn else np.full((8, 8), False)

        castling = np.asarray([
            np.full((8, 8), True) if board.has_queenside_castling_rights(chess.WHITE) else np.full((8, 8), False),
            np.full((8, 8), True) if board.has_kingside_castling_rights(chess.WHITE) else np.full((8, 8), False),
            np.full((8, 8), True) if board.has_queenside_castling_rights(chess.BLACK) else np.full((8, 8), False),
            np.full((8, 8), True) if board.has_kingside_castling_rights(chess.BLACK) else np.full((8, 8), False)
        ])

        counter = np.ones((8, 8)) if board.can_claim_fifty_moves() else np.zeros((8, 8))

        arrays = []
        for color in chess.COLORS:
            for piece in chess.PIECE_TYPES:
                array = np.full((8, 8), False)
                for index in list(board.pieces(piece, color)):
                    array[chess.square_rank(index), chess.square_file(index)] = True
                arrays.append(array)

        arrays = np.asarray(arrays)
        en_passant = np.full((8, 8), False)
        if board.has_legal_en_passant():
            assert board.ep_square is not None
            en_passant[chess.square_rank(board.ep_square), chess.square_file(board.ep_square)] = True

        r = np.array([turn, *castling, counter, *arrays, en_passant]).reshape(*config.INPUT_SHAPE)
        del board
        return r


    @staticmethod
    def estimate_winner(board: chess.Board) -> float:
        score = 0
        piece_values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9,
            chess.KING: 0
        }

        for piece in board.piece_map().values():
            if piece.color == chess.WHITE:
                score += piece_values[piece.piece_type]
            else:
                score -= piece_values[piece.piece_type]

        if np.abs(score) > 5:
            if score > 0:
                print("ESTIMATE: WHITE WINS")
                return 0.25
            else:
                print("ESTIMATE: BLACK WINS")
                return -0.25
        else:
            print("ESTIMATE: DRAW")
            return 0.0


    @staticmethod
    def get_n_pieces(board: chess.Board) -> int:
        return len(board.piece_map().values())


    def __str__(self):
        return str(self.board)
