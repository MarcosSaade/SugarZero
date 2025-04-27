import math
import random
import copy
from constants import EXPLORATION_WEIGHT
from game_state import GameState
from typing import List, Tuple

class MCTSNode:
    def __init__(self, game_state, parent=None, move=None):
        self.game_state = game_state
        self.parent = parent
        self.move = move  # Move that led to this state (start, end)
        self.children = []
        self.wins = 0
        self.visits = 0
        self.untried_moves = self.get_untried_moves()

    def get_untried_moves(self) -> List[Tuple[int, int]]:
        """Get all valid moves that haven't been tried yet."""
        return self.game_state.get_valid_moves()

    def add_child(self, move: Tuple[int, int]) -> 'MCTSNode':
        """Add a child node with the given move."""
        game_copy = copy.deepcopy(self.game_state)
        game_copy.make_move(move[0], move[1])
        child = MCTSNode(game_copy, parent=self, move=move)
        self.untried_moves.remove(move)
        self.children.append(child)
        return child

    def select_child(self) -> 'MCTSNode':
        """Select a child node using UCB1 formula."""
        return max(self.children, key=lambda c: c.ucb_value())

    def ucb_value(self) -> float:
        """Calculate UCB1 value for this node."""
        if self.visits == 0:
            return float('inf')
        exploitation = self.wins / self.visits
        exploration = EXPLORATION_WEIGHT * math.sqrt(math.log(self.parent.visits) / self.visits)
        return exploitation + exploration

    def is_fully_expanded(self) -> bool:
        """Check if all possible moves from this state have been tried."""
        return len(self.untried_moves) == 0

    def is_terminal(self) -> bool:
        """Check if this node represents a terminal game state."""
        return self.game_state.game_over

def mcts_search(game_state: GameState, ai_player: int, sim_count: int) -> Tuple[int, int]:
    """Execute Monte Carlo Tree Search to find the best move."""
    # Create root node
    root = MCTSNode(game_state)

    # Run simulations
    for _ in range(sim_count):
        # Phase 1: Selection
        node = root
        while node.is_fully_expanded() and not node.is_terminal():
            node = node.select_child()

        # Phase 2: Expansion
        if not node.is_terminal() and not node.is_fully_expanded():
            if node.untried_moves:  # Make sure there are moves to try
                move = random.choice(node.untried_moves)
                node = node.add_child(move)

        # Phase 3: Simulation (random playout)
        sim_state = copy.deepcopy(node.game_state)
        while not sim_state.game_over:
            valid_moves = sim_state.get_valid_moves()
            if not valid_moves:
                break
            start, end = random.choice(valid_moves)
            sim_state.make_move(start, end)

        # Phase 4: Backpropagation
        winner = sim_state.winner
        while node is not None:
            node.visits += 1
            # Add win if AI player won
            if winner == ai_player:
                node.wins += 1
            node = node.parent

    # Choose best move based on visit count
    if root.children:
        best_child = max(root.children, key=lambda c: c.visits)
        return best_child.move
    else:
        # Fallback to random valid move if no children (shouldn't happen)
        valid_moves = game_state.get_valid_moves()
        if valid_moves:
            return random.choice(valid_moves)
        else:
            return None