import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
import sys
import os
import threading
from queue import Queue

# Import mock pygame
import pygame
sys.modules['pygame'] = pygame

# Now import the game module that uses pygame
from game import Game

class TestGame(unittest.TestCase):
    def setUp(self):
        # Create a game instance with AI disabled for more deterministic testing
        self.game = Game(ai_enabled=False)
    
    def test_initialization(self):
        """Test that game initializes correctly"""
        self.assertEqual(self.game.turn, 1)  # Blue's turn
        self.assertEqual(self.game.move_count, 0)
        self.assertEqual(len(self.game.tiles), 9)  # 9 tiles on the board
        self.assertFalse(self.game.game_over)
        self.assertIsNone(self.game.winner)
        self.assertIsNone(self.game.selected)
    
    def test_mouse_to_square(self):
        """Test mouse position to board square mapping"""
        # Test valid square clicks
        self.assertEqual(self.game.mouse_to_square(100, 100), 0)  # Top-left
        self.assertEqual(self.game.mouse_to_square(300, 100), 1)  # Top-middle
        self.assertEqual(self.game.mouse_to_square(500, 100), 2)  # Top-right
        self.assertEqual(self.game.mouse_to_square(100, 300), 3)  # Middle-left
        self.assertEqual(self.game.mouse_to_square(300, 300), 4)  # Center
        
        # Test invalid clicks
        self.assertEqual(self.game.mouse_to_square(-10, 100), -1)  # Outside left
        self.assertEqual(self.game.mouse_to_square(100, 650), -1)  # In status bar
        self.assertEqual(self.game.mouse_to_square(700, 100), -1)  # Outside right
    
    def test_handle_click(self):
        """Test click handling for piece selection and movement"""
        # First click on blue piece
        self.game.handle_click((100, 500))  # Square 6 (blue piece)
        self.assertEqual(self.game.selected, 6)
        
        # Second click on valid move target
        self.game.handle_click((100, 300))  # Square 3 (empty square)
        self.assertIsNone(self.game.selected)  # Selection cleared
        self.assertEqual(self.game.board[3][0], 1)  # Blue piece at square 3
        self.assertEqual(self.game.board[6][1], -1)  # Piece moved from square 6
    
    def test_handle_keypress(self):
        """Test keyboard handling"""
        # Test reset key
        self.game.make_move(6, 3)  # Make a move
        self.assertEqual(self.game.move_count, 1)
        self.game.handle_keypress(pygame.K_r)  # Reset
        self.assertEqual(self.game.move_count, 0)  # Game state reset
        
        # Test AI toggle
        self.assertFalse(self.game.ai_enabled)  # AI disabled in setUp
        self.game.handle_keypress(pygame.K_a)  # Toggle AI
        self.assertTrue(self.game.ai_enabled)
        
        # Test AI player switch
        self.assertEqual(self.game.ai_player, 0)  # Default AI player is red
        self.game.handle_keypress(pygame.K_s)  # Switch AI player
        self.assertEqual(self.game.ai_player, 1)  # Now blue
        
        # Test difficulty cycle
        self.assertEqual(self.game.ai_difficulty, "Medium")  # Default
        self.game.handle_keypress(pygame.K_d)  # Cycle difficulty
        self.assertEqual(self.game.ai_difficulty, "Hard")
        self.game.handle_keypress(pygame.K_d)  # Cycle again
        self.assertEqual(self.game.ai_difficulty, "Easy")
        
        # Test quit key
        result = self.game.handle_keypress(pygame.K_q)
        self.assertFalse(result)  # Should return False to stop game loop
    
    def test_ai_move(self):
        """Test AI move generation"""
        # Enable AI for this test
        self.game.ai_enabled = True
        self.game.ai_player = 1  # Blue (current turn)
        self.game.ai_difficulty = "Easy"  # Faster search
        
        # Create a mock for the MCTS result
        mock_result = (6, 3)  # Move from square 6 to 3
        self.game.ai_result_queue.put(mock_result)
        
        # Trigger AI turn
        self.game.ai_turn()
        
        # Check that AI made the move
        self.assertEqual(self.game.board[3][0], 1)  # Blue piece at square 3
        self.assertEqual(self.game.turn, 0)  # Now red's turn
    
    def test_game_over_detection(self):
        """Test game over detection"""
        # Set up a board where red has no valid moves
        self.game.turn = 0  # Red's turn
        # Remove all red pieces
        for i in range(3):
            for j in range(6):
                if self.game.board[i][j] == 0:
                    self.game.board[i][j] = -1
        
        # Make blue pieces block all possible moves for red
        self.game.board[0][0] = 1
        self.game.board[1][0] = 1
        self.game.board[2][0] = 1
        
        # Check game over
        self.assertTrue(self.game.check_game_over())
        
        # Make a move to trigger game over state
        self.game.make_move(6, 3)  # This should not actually make a move now
        self.assertTrue(self.game.game_over)
        self.assertEqual(self.game.winner, 1)  # Blue wins

if __name__ == "__main__":
    unittest.main()