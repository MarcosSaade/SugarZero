def get_top_piece(board, square_idx):
    """Find the index of the topmost piece in a square."""
    square = board[square_idx]
    # Find the topmost piece
    for i in range(len(square)):
        if square[i] == -1:
            if i > 0:
                return i - 1  # Return index of top piece
            else:
                return -1  # Empty stack
    # If full stack, return the top
    return len(square) - 1

def get_top_empty(board, square_idx):
    """Find the index of the first empty space in a square."""
    square = board[square_idx]
    # Find the first empty space
    for i in range(len(square)):
        if square[i] == -1:
            return i  # Return index of empty spot
    # If full stack, return -1
    return -1

def is_forward_move(turn, start, end):
    """Check if a move is forward for the current player."""
    # For red (player 0), forward is moving from top to bottom
    if turn == 0:
        start_row = start // 3
        end_row = end // 3
        return end_row > start_row
    # For blue (player 1), forward is moving from bottom to top
    else:
        start_row = start // 3
        end_row = end // 3
        return end_row < start_row