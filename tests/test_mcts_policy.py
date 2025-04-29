# tests/test_mcts_policy.py
import sys, os, unittest
import torch
from game_state import GameState
from policy_net import PolicyNet
from value_net import ValueNet
from mcts import puct_search_with_policy
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class TestPUCTWithPolicy(unittest.TestCase):
    def setUp(self):
        self.game = GameState()
        self.policy = PolicyNet()
        self.value = ValueNet()

    def test_puct_search_with_policy(self):
        move, policy_dist = puct_search_with_policy(
            self.game, sim_count=5,
            policy_model=self.policy, value_model=self.value,
            cpuct=1.0, device='cpu'
        )
        # Move is a legal move
        self.assertIn(move, self.game.get_valid_moves())
        # Distribution keys match children moves
        self.assertTrue(abs(sum(policy_dist.values()) - 1.0) < 1e-5)
        for m,p in policy_dist.items():
            self.assertIsInstance(m, tuple)
            self.assertIsInstance(p, float)

if __name__ == "__main__":
    unittest.main()
