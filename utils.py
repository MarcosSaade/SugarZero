# utils.py

def get_top_piece(board, square_idx):
    """Find the index of the topmost piece in a square."""
    square = board[square_idx]
    for i, v in enumerate(square):
        if v == -1:
            return i - 1 if i > 0 else -1
    # If no empty spot, stack is full
    return len(square) - 1

def get_top_empty(board, square_idx):
    """Find the index of the first empty space in a square."""
    square = board[square_idx]
    for i, v in enumerate(square):
        if v == -1:
            return i
    # If no empty spot, return -1
    return -1

def is_forward_move(turn, start, end):
    """Check if a move is forward for the current player."""
    start_row = start // 3
    end_row = end // 3
    if turn == 0:
        # For red (player 0), forward is moving from top to bottom
        return end_row > start_row
    else:
        # For blue (player 1), forward is moving from bottom to top
        return end_row < start_row
