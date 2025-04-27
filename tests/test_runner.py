import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
import sys

# Import all test modules
from test_utils import TestUtils
from test_game_state import TestGameState
from test_mcts import TestMCTS
from test_integration import TestIntegration

def run_tests():
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTest(unittest.makeSuite(TestUtils))
    test_suite.addTest(unittest.makeSuite(TestGameState))
    test_suite.addTest(unittest.makeSuite(TestMCTS))
    test_suite.addTest(unittest.makeSuite(TestIntegration))
    
    # Run the tests
    result = unittest.TextTestRunner(verbosity=2).run(test_suite)
    
    # Return success status (0 if all tests passed, 1 otherwise)
    return 0 if result.wasSuccessful() else 1

if __name__ == "__main__":
    sys.exit(run_tests())