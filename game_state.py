import copy
from typing import List, Tuple
from constants import ADJACENT_SQUARES
from utils import get_top_piece, get_top_empty, is_forward_move

class GameState:
    def __init__(self, board=None, turn=None, move_count=None, moved_forward=None):
        if board:
            self.board = copy.deepcopy(board)
        else:
            # Initialize with default board
            self.board = [
                [0, 0, -1, -1, -1, -1],  # Two red pieces in square 0
                [0, 0, -1, -1, -1, -1],  # Two red pieces in square 1
                [0, 0, -1, -1, -1, -1],  # Two red pieces in square 2
                [-1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1],
                [1, 1, -1, -1, -1, -1],  # Two blue pieces in square 6
                [1, 1, -1, -1, -1, -1],  # Two blue pieces in square 7
                [1, 1, -1, -1, -1, -1],  # Two blue pieces in square 8
            ]

        self.turn = 1 if turn is None else turn
        self.move_count = 0 if move_count is None else move_count
        self.moved_forward = {0: False, 1: False} if moved_forward is None else copy.deepcopy(moved_forward)
        self.game_over = False
        self.winner = None

    def get_top_piece(self, square_idx):
        """Find index of topmost piece in a square."""
        return get_top_piece(self.board, square_idx)

    def get_top_empty(self, square_idx):
        """Find index of first empty space in a square."""
        return get_top_empty(self.board, square_idx)

    def is_forward_move(self, start, end):
        """Check if a move is forward for current player."""
        return is_forward_move(self.turn, start, end)

    def is_valid_move(self, start, end):
        """Check if a move is valid."""
        # Can't move to the same square
        if start == end:
            return False

        # Check if start has any pieces
        top_start = self.get_top_piece(start)
        if top_start == -1:
            return False

        # Check if the top piece belongs to the current player
        if self.board[start][top_start] != self.turn:
            return False

        # Check if end has space
        top_end = self.get_top_empty(end)
        if top_end == -1:
            return False

        # Check for adjacent squares (no diagonal movement)
        if end not in ADJACENT_SQUARES[start]:
            return False

        # Check height constraint (can only move to equal or lower stacks)
        start_height = sum(1 for piece in self.board[start] if piece != -1)
        end_height = sum(1 for piece in self.board[end] if piece != -1)
        if start_height < end_height:
            return False

        # Check forward movement requirement within first 3 moves per player
        player_move_count = self.move_count // 2 if self.turn == 0 else (self.move_count - 1) // 2
        if player_move_count < 3 and not self.moved_forward[self.turn]:
            # If this is one of the first 3 moves and player hasn't moved forward yet
            if self.is_forward_move(start, end):
                return True  # Valid forward move
            elif player_move_count == 2:  # Last chance to move forward
                return False  # Must move forward on the last chance
            # Otherwise, allow non-forward moves for first 2 moves

        return True

    def get_valid_moves(self) -> List[Tuple[int, int]]:
        """Get all valid moves for the current player."""
        valid_moves = []
        for start in range(9):
            top_piece = self.get_top_piece(start)
            if top_piece != -1 and self.board[start][top_piece] == self.turn:
                for end in range(9):
                    if self.is_valid_move(start, end):
                        valid_moves.append((start, end))
        return valid_moves

    def make_move(self, start, end):
        """Make a move on the board."""
        top_start = self.get_top_piece(start)
        top_end = self.get_top_empty(end)

        # Move the piece
        self.board[end][top_end] = self.board[start][top_start]
        self.board[start][top_start] = -1

        # Update forward movement tracking
        if self.is_forward_move(start, end):
            self.moved_forward[self.turn] = True

        # Update move count
        self.move_count += 1

        # Switch turns
        self.turn = 1 - self.turn

        # Check for game over
        if self.check_game_over():
            self.game_over = True
            self.winner = 1 - self.turn  # The previous player won

    def check_game_over(self):
        """Check if the game is over."""
        # Check if current player has any valid moves
        return len(self.get_valid_moves()) == 0