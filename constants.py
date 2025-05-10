# constants.py

import numpy as np

# Game constants
FPS = 60
WIDTH, HEIGHT = (600, 700)  # Increased height for status bar
TILE_SIZE = 200
STATUS_BAR_HEIGHT = 100

# Colors
GRAY_1 = (50, 50, 50)
GRAY_2 = (70, 70, 70)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
STATUS_BAR_COLOR = (30, 30, 30)

# Player colors
RED_COLORS  = [(200, 20, 0), (140, 20, 0)]
BLUE_COLORS = [(0, 20, 220), (0, 20, 160)]

# MCTS Constants
EXPLORATION_WEIGHT = 1.4  # UCB1 exploration parameter
MIN_MCTS_SIMULATIONS = 200  # Minimum number of simulations for MCTS

# Dirichlet‐noise parameters (for root)
DIRICHLET_ALPHA = 0.8 # inversely proportional to the number of moves, as seen in the AlphaZero paper
NOISE_EPSILON   = 0.1

# Sampling temperature (self-play)
# During the first N moves, sample with this temperature; afterwards, use greedy.
TEMP_MOVES_THRESHOLD = 3
TEMPERATURE           = 0.7

# AI difficulty settings
SIMULATION_COUNTS = {
    "Easy":   500,
    "Medium": 1000,
    "Hard":   5000
}

# Initial board setup as a numpy array (9 squares × stack height 6)
INITIAL_BOARD = np.array([
    # Opponent's pieces (top row - squares 0, 1, 2)
    [0, 0, -1, -1, -1, -1],
    [0, 0, -1, -1, -1, -1],
    [0, 0, -1, -1, -1, -1],

    # Middle row - squares 3, 4, 5
    [-1, -1, -1, -1, -1, -1],
    [-1, -1, -1, -1, -1, -1],
    [-1, -1, -1, -1, -1, -1],

    # Player's pieces (bottom row - squares 6, 7, 8)
    [1, 1, -1, -1, -1, -1],
    [1, 1, -1, -1, -1, -1],
    [1, 1, -1, -1, -1, -1],
], dtype=int)

# Square adjacency mapping
ADJACENT_SQUARES = {
    0: [1, 3],
    1: [0, 2, 4],
    2: [1, 5],
    3: [0, 4, 6],
    4: [1, 3, 5, 7],
    5: [2, 4, 8],
    6: [3, 7],
    7: [4, 6, 8],
    8: [5, 7]
}
