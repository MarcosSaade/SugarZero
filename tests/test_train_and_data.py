# tests/test_train_and_data.py
import sys, os, unittest
import torch
from replay_buffer import ReplayBuffer
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from train import scheduled_simulations, generate_self_play_data, train_step
from policy_net import PolicyNet
from value_net import ValueNet

class TestTrainModule(unittest.TestCase):
    def test_scheduled_simulations_extremes(self):
        # Should interpolate from MIN_MCTS_SIMULATIONS=100 to MCTS_SIMULATIONS=400
        self.assertEqual(scheduled_simulations(1), 100)
        self.assertEqual(scheduled_simulations(1000), 400)
        # Midpoint approximately halfway
        mid = scheduled_simulations(500)
        self.assertTrue(100 < mid < 400)

    def test_generate_self_play_data_structure(self):
        policy = PolicyNet()
        value = ValueNet()
        data = generate_self_play_data(policy, value, sim_count=2)
        # Must be non-empty
        self.assertGreater(len(data), 0)
        # Each entry is (Tensor, float, Tensor, float)
        state, turn, policy_t, value_t = data[0]
        self.assertIsInstance(state, torch.Tensor)
        self.assertIsInstance(turn, float)
        self.assertIsInstance(policy_t, torch.Tensor)
        self.assertIsInstance(value_t, float)
        # Policy tensor sums to 1
        self.assertAlmostEqual(policy_t.sum().item(), 1.0, places=5)

    def test_train_step_none_and_loss(self):
        # When buffer too small
        pb = PolicyNet()
        vb = ValueNet()
        buf = ReplayBuffer(capacity=5)
        loss = train_step(pb, vb, torch.optim.Adam(list(pb.parameters())+list(vb.parameters())), buf)
        self.assertIsNone(loss)
        # Fill buffer to batch size
        for _ in range(64):
            state = torch.zeros((2,3,3,6))
            buf.push(state, 0.0, torch.ones(81)/81, 1.0)
        loss2 = train_step(pb, vb, torch.optim.Adam(list(pb.parameters())+list(vb.parameters())), buf)
        self.assertIsInstance(loss2, float)
        self.assertGreaterEqual(loss2, 0.0)

if __name__ == "__main__":
    unittest.main()
