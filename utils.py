# utils.py
import numpy as np

def get_top_piece(board, square_idx):
    """Find the index of the topmost piece in a square."""
    square = board[square_idx]
    for i in range(len(square)):
        if square[i] == -1:
            return i - 1 if i > 0 else -1
    return len(square) - 1

def get_top_empty(board, square_idx):
    """Find the index of the first empty space in a square."""
    square = board[square_idx]
    for i in range(len(square)):
        if square[i] == -1:
            return i
    return -1

def is_forward_move(turn, start, end):
    """Check if a move is forward for the current player."""
    start_row, end_row = start // 3, end // 3
    return (end_row > start_row) if turn == 0 else (end_row < start_row)

def encode_board(board, turn):
    """
    Encode the board and current turn into numpy arrays suitable for a neural network.
    Channels: 0=red,1=blue; shape (2,3,3,6)
    Returns (tensor, turn).
    """
    tensor = np.zeros((2, 3, 3, 6), dtype=np.float32)
    for idx, square in enumerate(board):
        r, c = divmod(idx, 3)
        for h, p in enumerate(square):
            if p in (0, 1):
                tensor[p, r, c, h] = 1.0
    return tensor, turn

def move_to_index(move: tuple[int, int]) -> int:
    """Flatten a (start,end) into index in [0,80]: start*9 + end."""
    start, end = move
    return start * 9 + end

def index_to_move(idx: int) -> tuple[int, int]:
    """Convert flat index back to (start,end)."""
    return divmod(idx, 9)

def dirichlet_noise(priors: dict[tuple[int,int], float], alpha: float, epsilon: float) -> dict:
    """
    Add Dirichlet noise to priors dict at root:
      new_pi = (1 - epsilon)*pi + epsilon * eta,
    where eta ~ Dirichlet(alpha).
    """
    moves = list(priors.keys())
    probs = np.array([priors[m] for m in moves], dtype=np.float32)
    # sample noise
    eta = np.random.dirichlet([alpha] * len(moves))
    mixed = (1 - epsilon) * probs + epsilon * eta
    # renormalize
    mixed /= mixed.sum()
    return {m: float(mixed[i]) for i, m in enumerate(moves)}
