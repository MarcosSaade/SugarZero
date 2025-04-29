# tests/test_runner.py
import sys
import os
import unittest
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import each TestCase
from test_utils import TestUtils
from test_game_state import TestGameState
from test_game import TestGame
from test_mcts import TestPUCT
from test_utils import TestUtils        # merged extended utils
from test_replaybuffer import TestReplayBuffer
from test_networks import TestNetworks
from test_train_and_data import TestTrainModule
from test_mcts_policy import TestPUCTWithPolicy
from test_integration import TestIntegration

if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    suite.addTests(loader.loadTestsFromTestCase(TestUtils))
    suite.addTests(loader.loadTestsFromTestCase(TestGameState))
    suite.addTests(loader.loadTestsFromTestCase(TestGame))
    suite.addTests(loader.loadTestsFromTestCase(TestPUCT))
    suite.addTests(loader.loadTestsFromTestCase(TestReplayBuffer))
    suite.addTests(loader.loadTestsFromTestCase(TestNetworks))
    suite.addTests(loader.loadTestsFromTestCase(TestTrainModule))
    suite.addTests(loader.loadTestsFromTestCase(TestPUCTWithPolicy))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(not result.wasSuccessful())
