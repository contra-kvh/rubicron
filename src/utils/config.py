import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
SOCKET_HOST = "localhost"
SOCKET_PORT = 5000
SOCKET_BUFFER_SIZE = 8192

SIMULATIONS_PER_MOVE = 200
C_BASE = 20000
C_INIT = 2

DIRICHLET_NOISE = 0.4
MAX_GAME_MOVES = 200


# --- HYPERPARAMETERS --- #
LEARNING_RATE = 0.2
CONV_FILTERS = 256
N_RESIDUAL_BLOCKS = 20
N_EPOCHS = 5

BATCH_SIZE = 64
LOSS_PLOTS_FOLDER = "../data/plots"
MEM_DIR = "../data/memory"
MODEL_DIR = "../data/models"
SL_MODEL_PATH = "../data/models/sl.pt"
MAX_REPLAY_MEMORY = 100000000


# --- NN CONFIGURATION --- #
n = 8
_n_input_planes = (2 * 6 + 1) + (1 + 4 + 1)
INPUT_SHAPE = (_n_input_planes, n, n)

_queen_planes = 56
_knight_planes = 8
_underpromotion_planes = 9
n_output_planes = _queen_planes + _knight_planes + _underpromotion_planes
OUTPUT_SHAPE = (8*8*n_output_planes, 1)
