# tests/test_utils.py
import sys
import os
import unittest
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import (
    get_top_piece,
    get_top_empty,
    is_forward_move,
    encode_board,
    move_to_index,
    index_to_move
)

class TestUtils(unittest.TestCase):
    def test_get_top_piece(self):
        board = [[-1, -1, -1, -1, -1, -1]]
        self.assertEqual(get_top_piece(board, 0), -1)
        board = [[0, -1, -1, -1, -1, -1]]
        self.assertEqual(get_top_piece(board, 0), 0)
        board = [[0, 1, -1, -1, -1, -1]]
        self.assertEqual(get_top_piece(board, 0), 1)
        board = [[0, 1, 0, 1, 0, 1]]
        self.assertEqual(get_top_piece(board, 0), 5)

    def test_get_top_empty(self):
        board = [[-1, -1, -1, -1, -1, -1]]
        self.assertEqual(get_top_empty(board, 0), 0)
        board = [[0, -1, -1, -1, -1, -1]]
        self.assertEqual(get_top_empty(board, 0), 1)
        board = [[0, 1, -1, -1, -1, -1]]
        self.assertEqual(get_top_empty(board, 0), 2)
        board = [[0, 1, 0, 1, 0, 1]]
        self.assertEqual(get_top_empty(board, 0), -1)

    def test_is_forward_move(self):
        # Red (0) moves down
        self.assertTrue(is_forward_move(0, 0, 3))
        self.assertFalse(is_forward_move(0, 3, 0))
        # Blue (1) moves up
        self.assertTrue(is_forward_move(1, 6, 3))
        self.assertFalse(is_forward_move(1, 3, 6))

    def test_encode_board_shape_and_counts(self):
        board = [[-1]*6 for _ in range(9)]
        board[0][0] = 0
        board[8][0] = 1
        tensor, turn = encode_board(board, turn=1)
        self.assertEqual(tensor.shape, (2,3,3,6))
        self.assertEqual(tensor[0,0,0,0], 1.0)
        self.assertEqual(tensor[1,2,2,0], 1.0)
        self.assertEqual(float(tensor.sum()), 2.0)
        self.assertIn(turn, (0,1))

    def test_move_index_roundtrip(self):
        for start in [0,4,8]:
            for end in [0,3,5,8]:
                idx = move_to_index((start,end))
                s2, e2 = index_to_move(idx)
                self.assertEqual((s2,e2), (start,end))

if __name__ == "__main__":
    unittest.main()
