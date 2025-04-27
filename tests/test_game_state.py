import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
from game_state import GameState

class TestGameState(unittest.TestCase):
    def setUp(self):
        # Setup a clean game state before each test
        self.game_state = GameState()
    
    def test_initialization(self):
        # Test default initialization
        self.assertEqual(self.game_state.turn, 1)
        self.assertEqual(self.game_state.move_count, 0)
        self.assertEqual(self.game_state.moved_forward, {0: False, 1: False})
        self.assertFalse(self.game_state.game_over)
        self.assertIsNone(self.game_state.winner)
        
        # Test that board is initialized correctly
        # Check that red pieces are in the top row
        for i in range(3):
            self.assertEqual(self.game_state.board[i][0], 0)
            self.assertEqual(self.game_state.board[i][1], 0)
        
        # Check that blue pieces are in the bottom row
        for i in range(6, 9):
            self.assertEqual(self.game_state.board[i][0], 1)
            self.assertEqual(self.game_state.board[i][1], 1)
    
    def test_is_valid_move(self):
        # Test same square
        self.assertFalse(self.game_state.is_valid_move(0, 0))
        
        # Test empty source square
        empty_board = [[]]
        self.game_state.board = empty_board * 9  # Make all squares empty
        self.assertFalse(self.game_state.is_valid_move(0, 1))
        
        # Reset board
        self.setUp()
        
        # Test wrong player's piece
        self.game_state.turn = 0  # Red's turn
        self.assertFalse(self.game_state.is_valid_move(6, 3))  # Blue piece at 6
        
        # Test non-adjacent square
        self.game_state.turn = 0  # Red's turn
        self.assertFalse(self.game_state.is_valid_move(0, 2))  # 0 to 2 (non-adjacent)
        
        # Test valid move
        self.game_state.turn = 0  # Red's turn
        self.assertTrue(self.game_state.is_valid_move(0, 1))  # 0 to 1 (adjacent)
        
        # Test height constraint
        # Make stack 0 shorter than stack 1
        self.game_state.board[0][0] = -1  # Remove a piece from 0
        self.game_state.board[1][2] = 0   # Add a piece to 1
        self.assertFalse(self.game_state.is_valid_move(0, 1))
    
    def test_get_valid_moves(self):
        # Blue's turn
        valid_moves = self.game_state.get_valid_moves()
        self.assertIn((6, 3), valid_moves)  # 6 to 3 is valid
        self.assertIn((7, 4), valid_moves)  # 7 to 4 is valid
        self.assertIn((8, 5), valid_moves)  # 8 to 5 is valid
        
        # Red's turn
        self.game_state.turn = 0
        valid_moves = self.game_state.get_valid_moves()
        self.assertIn((0, 1), valid_moves)  # 0 to 1 is valid
        self.assertIn((1, 0), valid_moves)  # 1 to 0 is valid
        self.assertIn((1, 2), valid_moves)  # 1 to 2 is valid
    
    def test_make_move(self):
        # Test making a valid move for blue
        self.game_state.make_move(6, 3)
        
        # Check that piece moved correctly
        self.assertEqual(self.game_state.board[6][0], 1)  # One piece still at 6
        self.assertEqual(self.game_state.board[6][1], -1)  # Empty spot at 6
        self.assertEqual(self.game_state.board[3][0], 1)  # Blue piece now at 3
        
        # Check that turn switched to red
        self.assertEqual(self.game_state.turn, 0)
        
        # Check move count increased
        self.assertEqual(self.game_state.move_count, 1)
        
        # Check forward move tracking
        self.assertTrue(self.game_state.moved_forward[1])  # Blue moved forward
    
    def test_check_game_over(self):
        # Set up a board where red has no valid moves
        self.game_state.turn = 0
        # Remove all red pieces
        for i in range(3):
            for j in range(6):
                if self.game_state.board[i][j] == 0:
                    self.game_state.board[i][j] = -1
        
        # Game should be over
        self.assertTrue(self.game_state.check_game_over())
        
        # Reset and set up normal board
        self.setUp()
        
        # Game should not be over
        self.assertFalse(self.game_state.check_game_over())

if __name__ == "__main__":
    unittest.main()