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
RED_COLORS = [(200, 20, 0), (170, 20, 0)]
BLUE_COLORS = [(0, 20, 220), (0, 20, 190)]

# MCTS Constants
EXPLORATION_WEIGHT = 1.4  # UCB1 exploration parameter

# AI difficulty settings
SIMULATION_COUNTS = {
    "Easy": 50,
    "Medium": 100,
    "Hard": 250
}

# Initial board setup
INITIAL_BOARD = [
    # Opponent's pieces (top row - squares 0, 1, 2)
    [0, 0, -1, -1, -1, -1],  # Two red pieces in square 0
    [0, 0, -1, -1, -1, -1],  # Two red pieces in square 1
    [0, 0, -1, -1, -1, -1],  # Two red pieces in square 2

    # Middle row - squares 3, 4, 5
    [-1, -1, -1, -1, -1, -1],
    [-1, -1, -1, -1, -1, -1],
    [-1, -1, -1, -1, -1, -1],

    # Player's pieces (bottom row - squares 6, 7, 8)
    [1, 1, -1, -1, -1, -1],  # Two blue pieces in square 6
    [1, 1, -1, -1, -1, -1],  # Two blue pieces in square 7
    [1, 1, -1, -1, -1, -1],  # Two blue pieces in square 8
]

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