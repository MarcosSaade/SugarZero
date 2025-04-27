# game_state.py

from typing import List, Tuple
from constants import ADJACENT_SQUARES
from utils import get_top_piece, get_top_empty, is_forward_move

class GameState:
    def __init__(self, board=None, turn=None, move_count=None, moved_forward=None):
        if board is not None:
            # Deep‐copy board rows
            self.board = [row.copy() for row in board]
        else:
            # Default starting layout
            self.board = [
                [0, 0, -1, -1, -1, -1],
                [0, 0, -1, -1, -1, -1],
                [0, 0, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1],
                [1, 1, -1, -1, -1, -1],
                [1, 1, -1, -1, -1, -1],
                [1, 1, -1, -1, -1, -1],
            ]

        self.turn          = 1 if turn is None else turn
        self.move_count    = 0 if move_count is None else move_count
        self.moved_forward = {0: False, 1: False} if moved_forward is None else moved_forward.copy()
        self.game_over     = False
        self.winner        = None

    def clone(self) -> 'GameState':
        """Fast clone for MCTS (avoids full deepcopy)."""
        new = object.__new__(GameState)
        new.board          = [row.copy() for row in self.board]
        new.turn           = self.turn
        new.move_count     = self.move_count
        new.moved_forward  = self.moved_forward.copy()
        new.game_over      = self.game_over
        new.winner         = self.winner
        return new

    def get_top_piece(self, square_idx):
        return get_top_piece(self.board, square_idx)

    def get_top_empty(self, square_idx):
        return get_top_empty(self.board, square_idx)

    def is_forward_move(self, start, end):
        return is_forward_move(self.turn, start, end)

    def is_valid_move(self, start, end):
        if start == end:
            return False
        ts = self.get_top_piece(start)
        if ts == -1 or self.board[start][ts] != self.turn:
            return False
        te = self.get_top_empty(end)
        if te == -1 or end not in ADJACENT_SQUARES[start]:
            return False
        # Height constraint
        h_s = sum(1 for p in self.board[start] if p != -1)
        h_e = sum(1 for p in self.board[end]   if p != -1)
        if h_s < h_e:
            return False
        # Forward‐move requirement in first 3 moves
        pmc = (self.move_count // 2) if self.turn == 0 else ((self.move_count - 1) // 2)
        if pmc < 3 and not self.moved_forward[self.turn]:
            if self.is_forward_move(start, end):
                return True
            if pmc == 2:
                return False
        return True

    def get_valid_moves(self) -> List[Tuple[int, int]]:
        moves = []
        for s in range(len(self.board)):
            tp = self.get_top_piece(s)
            if tp != -1 and self.board[s][tp] == self.turn:
                for e in range(len(self.board)):
                    if self.is_valid_move(s, e):
                        moves.append((s, e))
        return moves

    def make_move(self, start, end):
        ts = self.get_top_piece(start)
        te = self.get_top_empty(end)
        self.board[end][te]   = self.board[start][ts]
        self.board[start][ts] = -1

        if self.is_forward_move(start, end):
            self.moved_forward[self.turn] = True

        self.move_count += 1
        self.turn       = 1 - self.turn

        if self.check_game_over():
            self.game_over = True
            self.winner    = 1 - self.turn

    def check_game_over(self) -> bool:
        """Return True if no valid moves remain for current player."""
        return len(self.get_valid_moves()) == 0
