# tests/test_replaybuffer.py
import sys, os, unittest
import torch
from collections import Counter
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from replay_buffer import ReplayBuffer

class TestReplayBuffer(unittest.TestCase):
    def test_push_and_len(self):
        buf = ReplayBuffer(capacity=3)
        self.assertEqual(len(buf), 0)
        # Push 4 items; capacity should cap at 3
        for i in range(4):
            state = torch.zeros((2,3,3,6))
            buf.push(state, float(i%2), torch.ones(81)/81, float((i+1)%2))
        self.assertEqual(len(buf), 3)

    def test_sample_shapes_and_types(self):
        buf = ReplayBuffer(capacity=10)
        for _ in range(8):
            state = torch.randn((2,3,3,6))
            buf.push(state, 0.0, torch.randn(81), 1.0)
        states, turns, policies, values = buf.sample(5)
        self.assertEqual(states.shape, (5,2,3,3,6))
        self.assertEqual(turns.shape, (5,1))
        self.assertEqual(policies.shape, (5,81))
        self.assertEqual(values.shape, (5,1))
        self.assertTrue(isinstance(states, torch.Tensor))

if __name__ == "__main__":
    unittest.main()
