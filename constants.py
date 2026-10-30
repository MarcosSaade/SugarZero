# constants.py
#
# Display and difficulty settings for the pygame interface.

FPS = 60
WIDTH, HEIGHT = 600, 700
TILE_SIZE = 200
STATUS_BAR_HEIGHT = 100

GRAY_1 = (50, 50, 50)
GRAY_2 = (70, 70, 70)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
STATUS_BAR_COLOR = (30, 30, 30)
SELECTED_COLOR = (100, 255, 100)
LAST_MOVE_COLOR = (255, 200, 0)

# Shades alternate by stack level so individual pieces stay distinguishable.
RED_COLORS = [(200, 20, 0), (140, 20, 0)]
BLUE_COLORS = [(0, 20, 220), (0, 20, 160)]

# MCTS simulations per AI move.
SIMULATION_COUNTS = {
    "Easy": 25,
    "Medium": 200,
    "Hard": 800,
}
DIFFICULTIES = list(SIMULATION_COUNTS)

MODEL_PATH = "models/sugarzero.pt"
