import chess
from chess import Move
import numpy as np
from PIL import Image
import time
from mcts.mapper import Mapping
from mcts.node import Node

def fen_to_input(fen: str) -> np.ndarray:
    board = chess.Board(fen)

    # Initialize 19 feature planes
    planes = []

    # Turn (1 plane)
    turn = np.full((8, 8), board.turn, dtype=np.float32)  # True for White, False for Black
    planes.append(turn)

    # Castling rights (4 planes)
    castling = [
        np.full((8, 8), board.has_queenside_castling_rights(chess.WHITE), dtype=np.float32),
        np.full((8, 8), board.has_kingside_castling_rights(chess.WHITE), dtype=np.float32),
        np.full((8, 8), board.has_queenside_castling_rights(chess.BLACK), dtype=np.float32),
        np.full((8, 8), board.has_kingside_castling_rights(chess.BLACK), dtype=np.float32)
    ]
    planes.extend(castling)

    # Fifty-move counter (1 plane)
    counter = np.ones((8, 8), dtype=np.float32) if board.can_claim_fifty_moves() else np.zeros((8, 8), dtype=np.float32)
    planes.append(counter)

    # Piece positions (12 planes: 6 piece types × 2 colors)
    for color in chess.COLORS:
        for piece in chess.PIECE_TYPES:
            array = np.zeros((8, 8), dtype=np.float32)
            for index in board.pieces(piece, color):
                array[chess.square_rank(index), chess.square_file(index)] = 1
            planes.append(array)

    # En passant (1 plane)
    en_passant = np.zeros((8, 8), dtype=np.float32)
    if board.has_legal_en_passant() and board.ep_square is not None:
        en_passant[chess.square_rank(board.ep_square), chess.square_file(board.ep_square)] = 1
    planes.append(en_passant)

    # Stack planes and reshape to (19, 8, 8)
    r = np.stack(planes, axis=0)  # Shape: (19, 8, 8)
    del board
    return r


def get_height_of_tree(node: Node | None):
    if node is None:
        return 0
    h = 0
    for edge in node.edges:
        h = max(h, get_height_of_tree(edge.output_node))
    return h + 1


def move_to_plane_index(_move: str, board: chess.Board):
    move: Move = Move.from_uci(_move)
    # get start and end position
    from_square = move.from_square
    to_square = move.to_square

    # get piece
    piece: chess.Piece | None = board.piece_at(from_square)

    if piece is None:
            raise Exception(f"No piece at {from_square}")

    plane_index: int | None = None

    if move.promotion and move.promotion != chess.QUEEN:
        piece_type, direction = Mapping.get_underpromotion_move(
            move.promotion, from_square, to_square
        )
        plane_index = Mapping.mapper[piece_type][1 - direction]
    else:
        if piece.piece_type == chess.KNIGHT:
            # get direction
                direction = Mapping.get_knight_move(from_square, to_square)
                plane_index = Mapping.mapper[direction]
        else:
            # get direction of queen-type move
            direction, distance = Mapping.get_queenlike_move(
                from_square, to_square)
            plane_index = Mapping.mapper[direction][np.abs(distance)-1]

    row = chess.square_rank(from_square)
    col = chess.square_file(from_square)
    return (plane_index, row, col)


def time_function(func):
    """
    Decorator to time a function
    """
    def wrap_func(*args, **kwargs):
        t1 = time.time()
        result = func(*args, **kwargs)
        t2 = time.time()
        print(f'Function {func.__name__!r} executed in {(t2-t1):.4f}s')
        return result
    return wrap_func

def moves_to_output_vector(moves: dict, board: chess.Board) -> np.ndarray:
    """
    Convert a dictionary of moves to a vector of probabilities
    """
    vector = np.zeros((73, 8, 8), dtype=np.float32)
    for move in moves:
        plane_index, row, col = move_to_plane_index(move, board)
        vector[plane_index, row, col] = moves[move]
    return np.asarray(vector)


def save_output_state_to_imgs(output_state: np.ndarray, path: str, name: str = "full"):
    """
    Save an output state to images
    """
    start_time = time.time()
    # full image of all states
    # pad input_state with grey values
    output_state = np.pad(output_state.astype(float)*255, ((0, 0), (1, 1), (1, 1)), 'constant', constant_values=128)
    full_array = np.concatenate(output_state, axis=1)
    # more padding
    full_array = np.pad(full_array, ((4, 4), (5, 5)), 'constant', constant_values=128)
    img = Image.fromarray(full_array.astype(np.uint8))
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img.save(f"{path}/{name}.png")
    print(
        f"*** Saving to images: {(time.time() - start_time):.6f} seconds ***")


def save_input_state_to_imgs(input_state: np.ndarray, path: str):
    start_time = time.time()
    # full image of all states
    # convert booleans to integers
    input_state = np.array(input_state)*np.uint8(255)
    # pad input_state with grey values
    input_state = np.pad(input_state, ((0, 0), (1, 1), (1, 1)),
                         'constant', constant_values=128)

    full_array = np.concatenate(input_state, axis=1)
    # more padding
    full_array = np.pad(full_array, ((4, 4), (5, 5)),
                        'constant', constant_values=128)
    img = Image.fromarray(full_array)
    img.save(f"{path}/full.png")
    print(
        f"*** Saving to images: {(time.time() - start_time):.6f} seconds ***")
