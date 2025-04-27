import unittest
import math
from mcts import MCTSNode, mcts_search
from game_state import GameState

class TestMCTS(unittest.TestCase):
    def setUp(self):
        self.game_state = GameState()
        self.root_node = MCTSNode(self.game_state)
    
    def test_node_initialization(self):
        self.assertEqual(self.root_node.wins, 0)
        self.assertEqual(self.root_node.visits, 0)
        self.assertIsNone(self.root_node.parent)
        self.assertIsNone(self.root_node.move)
        self.assertGreater(len(self.root_node.untried_moves), 0)
    
    def test_add_child(self):
        # Get an untried move
        move = self.root_node.untried_moves[0]
        
        # Add child
        child = self.root_node.add_child(move)
        
        # Check that child was created correctly
        self.assertEqual(child.parent, self.root_node)
        self.assertEqual(child.move, move)
        self.assertEqual(child.wins, 0)
        self.assertEqual(child.visits, 0)
        self.assertIn(child, self.root_node.children)
        self.assertNotIn(move, self.root_node.untried_moves)
    
    def test_ucb_value(self):
        # Set up a parent node with visits
        parent = MCTSNode(self.game_state)
        parent.visits = 10
        
        # Create a child node
        child = MCTSNode(self.game_state, parent=parent)
        child.visits = 5
        child.wins = 3
        
        # Compute expected UCB value using actual formula
        exploitation = child.wins / child.visits
        exploration = math.sqrt(math.log(parent.visits) / child.visits)
        expected_value = exploitation + 1.4 * exploration
        
        # Allow a small tolerance for floating-point arithmetic
        self.assertAlmostEqual(child.ucb_value(), expected_value, places=2)
    
    def test_is_fully_expanded(self):
        # Initially not fully expanded
        self.assertFalse(self.root_node.is_fully_expanded())
        
        # Add all children
        while self.root_node.untried_moves:
            move = self.root_node.untried_moves[0]
            self.root_node.add_child(move)
        
        # Now fully expanded
        self.assertTrue(self.root_node.is_fully_expanded())
    
    def test_is_terminal(self):
        # Initial state is not terminal
        self.assertFalse(self.root_node.is_terminal())
        
        # Create a terminal game state
        terminal_state = GameState()
        terminal_state.game_over = True
        terminal_node = MCTSNode(terminal_state)
        
        # Should be terminal
        self.assertTrue(terminal_node.is_terminal())
    
    def test_mcts_search(self):
        # Test that mcts_search returns a valid move
        # Use a small simulation count for faster testing
        move = mcts_search(self.game_state, 0, 10)
        
        # Check that move is a tuple of two integers
        self.assertIsInstance(move, tuple)
        self.assertEqual(len(move), 2)
        self.assertIsInstance(move[0], int)
        self.assertIsInstance(move[1], int)
        
        # Check that move is valid
        valid_moves = self.game_state.get_valid_moves()
        self.assertIn(move, valid_moves)

if __name__ == "__main__":
    unittest.main()
