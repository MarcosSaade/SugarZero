import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
import copy
from game_state import GameState
from mcts import mcts_search

class TestIntegration(unittest.TestCase):
    def test_game_flow(self):
        """Test a complete game flow with several moves."""
        game = GameState()
        
        # Blue's turn first
        self.assertEqual(game.turn, 1)
        
        # Make a series of moves
        game.make_move(6, 3)  # Blue moves
        self.assertEqual(game.turn, 0)  # Red's turn now
        self.assertEqual(game.move_count, 1)
        self.assertTrue(game.moved_forward[1])  # Blue moved forward
        
        game.make_move(0, 3)  # Red moves
        self.assertEqual(game.turn, 1)  # Blue's turn now
        self.assertEqual(game.move_count, 2)
        self.assertTrue(game.moved_forward[0])  # Red moved forward
        
        # Check that pieces are in the expected positions
        self.assertEqual(game.board[3][0], 1)  # Blue piece at 3
        self.assertEqual(game.board[3][1], 0)  # Red piece at 3
        
        # Continue with a few more moves
        game.make_move(7, 4)  # Blue moves
        game.make_move(1, 4)  # Red moves
        
        # Verify game state is as expected
        self.assertEqual(game.turn, 1)  # Blue's turn
        self.assertEqual(game.move_count, 4)
        self.assertFalse(game.game_over)
    
    def test_forward_move_requirement(self):
        """Test the requirement to move forward at least once in first 3 moves."""
        game = GameState()
        
        # First move for blue - not forward
        valid_moves = game.get_valid_moves()
        horizontal_moves = [(6, 7), (7, 6), (7, 8), (8, 7)]
        valid_horizontal = [m for m in horizontal_moves if m in valid_moves]
        
        if valid_horizontal:
            game.make_move(valid_horizontal[0][0], valid_horizontal[0][1])
            self.assertFalse(game.moved_forward[1])  # Blue didn't move forward
            
            # First move for red - not forward
            game.make_move(0, 1)  # Horizontal move
            self.assertFalse(game.moved_forward[0])  # Red didn't move forward
            
            # Second move for blue - not forward
            game.make_move(valid_horizontal[0][1], valid_horizontal[0][0])  # Move back
            self.assertFalse(game.moved_forward[1])  # Blue still didn't move forward
            
            # Check that on third move, blue must move forward
            valid_moves = game.get_valid_moves()
            horizontal_moves = [(6, 7), (7, 6), (7, 8), (8, 7)]
            valid_horizontal = [m for m in horizontal_moves if m in valid_moves]
            
            # All horizontal moves should be invalid now
            for move in valid_horizontal:
                self.assertFalse(game.is_valid_move(move[0], move[1]))
    
    def test_mcts_integration(self):
        """Test that MCTS can find good moves in typical game situations."""
        # Test in initial position
        game = GameState()
        move = mcts_search(game, 1, 50)  # Blue to move
        self.assertIsNotNone(move)
        
        # Make sure move is valid
        valid_moves = game.get_valid_moves()
        self.assertIn(move, valid_moves)
        
        # Test in a mid-game position
        game.make_move(6, 3)  # Blue moves
        game.make_move(0, 3)  # Red moves
        game.make_move(7, 4)  # Blue moves
        game.make_move(1, 4)  # Red moves
        
        # Now test MCTS for blue
        move = mcts_search(game, 1, 50)
        self.assertIsNotNone(move)
        
        # Make sure move is valid
        valid_moves = game.get_valid_moves()
        self.assertIn(move, valid_moves)

if __name__ == "__main__":
    unittest.main()