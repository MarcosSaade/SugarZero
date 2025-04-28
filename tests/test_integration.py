# tests/test_integration.py

import sys
import os
import unittest
import torch

# allow imports from project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from game_state import GameState
from mcts import puct_search
from policy_net import PolicyNet
from value_net import ValueNet


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
            game.make_move(*valid_horizontal[0])
            self.assertFalse(game.moved_forward[1])  # Blue didn't move forward
            
            # First move for red - not forward
            game.make_move(0, 1)  # Horizontal move
            self.assertFalse(game.moved_forward[0])  # Red didn't move forward
            
            # Second move for blue - not forward
            back = valid_horizontal[0][::-1]
            game.make_move(*back)
            self.assertFalse(game.moved_forward[1])  # Blue still didn't move forward
            
            # On third blue move, horizontal should be invalid
            valid_moves = game.get_valid_moves()
            for move in horizontal_moves:
                if move in valid_moves:
                    self.assertFalse(game.is_valid_move(*move))
    
    def test_puct_integration(self):
        """Test that PUCT search returns a valid move."""
        game = GameState()
        
        # Dummy uniform policy
        policy = PolicyNet()
        def uniform_policy(board, turn):
            bsz = board.size(0)
            return torch.ones(bsz, 81) / 81
        policy.forward = uniform_policy
        
        # Dummy constant value
        value = ValueNet()
        def constant_value(board, turn):
            bsz = board.size(0)
            return torch.full((bsz, 1), 0.5)
        value.forward = constant_value
        
        # Run PUCT
        move = puct_search(
            game,
            sim_count=50,
            policy_model=policy,
            value_model=value,
            cpuct=1.4,
            device='cpu'
        )
        
        # Move must be one of the legal moves
        self.assertIsNotNone(move)
        self.assertIn(move, game.get_valid_moves())


if __name__ == "__main__":
    unittest.main()
