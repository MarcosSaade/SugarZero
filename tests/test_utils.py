import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
import sys
import os
from utils import get_top_piece, get_top_empty, is_forward_move
# Add the parent directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class TestUtils(unittest.TestCase):
    def test_get_top_piece(self):
        # Test empty square
        board = [[-1, -1, -1, -1, -1, -1]]
        self.assertEqual(get_top_piece(board, 0), -1)
        
        # Test one piece
        board = [[0, -1, -1, -1, -1, -1]]
        self.assertEqual(get_top_piece(board, 0), 0)
        
        # Test multiple pieces
        board = [[0, 1, -1, -1, -1, -1]]
        self.assertEqual(get_top_piece(board, 0), 1)
        
        # Test full stack
        board = [[0, 1, 0, 1, 0, 1]]
        self.assertEqual(get_top_piece(board, 0), 5)
    
    def test_get_top_empty(self):
        # Test empty square
        board = [[-1, -1, -1, -1, -1, -1]]
        self.assertEqual(get_top_empty(board, 0), 0)
        
        # Test one piece
        board = [[0, -1, -1, -1, -1, -1]]
        self.assertEqual(get_top_empty(board, 0), 1)
        
        # Test multiple pieces
        board = [[0, 1, -1, -1, -1, -1]]
        self.assertEqual(get_top_empty(board, 0), 2)
        
        # Test full stack
        board = [[0, 1, 0, 1, 0, 1]]
        self.assertEqual(get_top_empty(board, 0), -1)
    
    def test_is_forward_move(self):
        # For red player (0)
        # Forward is top to bottom (increasing row)
        self.assertTrue(is_forward_move(0, 0, 3))  # 0 (row 0) to 3 (row 1)
        self.assertTrue(is_forward_move(0, 1, 4))  # 1 (row 0) to 4 (row 1)
        self.assertFalse(is_forward_move(0, 3, 0))  # 3 (row 1) to 0 (row 0)
        self.assertFalse(is_forward_move(0, 4, 1))  # 4 (row 1) to 1 (row 0)
        
        # For blue player (1)
        # Forward is bottom to top (decreasing row)
        self.assertTrue(is_forward_move(1, 6, 3))  # 6 (row 2) to 3 (row 1)
        self.assertTrue(is_forward_move(1, 7, 4))  # 7 (row 2) to 4 (row 1)
        self.assertFalse(is_forward_move(1, 3, 6))  # 3 (row 1) to 6 (row 2)
        self.assertFalse(is_forward_move(1, 4, 7))  # 4 (row 1) to 7 (row 2)

if __name__ == "__main__":
    unittest.main()