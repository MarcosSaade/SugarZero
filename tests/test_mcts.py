# tests/test_mcts.py

import unittest
import math
import torch

from mcts import PUCTNode, puct_search
from game_state import GameState
from policy_net import PolicyNet
from value_net import ValueNet

class TestPUCT(unittest.TestCase):
    def setUp(self):
        self.game_state = GameState()

        # Dummy policy: uniform over all 81 moves
        self.policy_model = PolicyNet()
        def uniform_policy(board, turn):
            bsz = board.size(0)
            return torch.ones(bsz, 81) / 81
        self.policy_model.forward = uniform_policy

        # Dummy value: constant 0.5 for any state
        self.value_model = ValueNet()
        def constant_value(board, turn):
            bsz = board.size(0)
            return torch.full((bsz, 1), 0.5)
        self.value_model.forward = constant_value

    def test_puct_search_returns_valid_move(self):
        move = puct_search(
            self.game_state,
            sim_count=10,
            policy_model=self.policy_model,
            value_model=self.value_model,
            cpuct=1.0,
            device='cpu'
        )
        self.assertIsInstance(move, tuple)
        self.assertIn(move, self.game_state.get_valid_moves())


if __name__ == "__main__":
    unittest.main()
