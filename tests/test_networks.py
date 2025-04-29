# tests/test_networks.py
import sys, os, unittest
import torch
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from policy_net import PolicyNet
from value_net import ValueNet

class TestNetworks(unittest.TestCase):
    def setUp(self):
        self.batch = 4
        self.board = torch.randn((self.batch,2,3,3,6))
        self.turns = torch.randint(0,2,(self.batch,))

    def test_policy_net_output(self):
        net = PolicyNet()
        out = net(self.board, self.turns)
        self.assertEqual(out.shape, (self.batch,81))
        # each row sums to 1
        sums = out.sum(dim=1).tolist()
        for s in sums:
            self.assertAlmostEqual(s, 1.0, places=5)
        self.assertTrue((out >= 0).all())

    def test_value_net_output(self):
        net = ValueNet()
        out = net(self.board, self.turns)
        self.assertEqual(out.shape, (self.batch,1))
        # values in [0,1]
        self.assertTrue(((out >= 0) & (out <= 1)).all())

if __name__ == "__main__":
    unittest.main()
