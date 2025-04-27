# mcts.py

import math
import random
from constants import EXPLORATION_WEIGHT
from game_state import GameState
from typing import List, Tuple

class MCTSNode:
    def __init__(self, game_state: GameState, parent=None, move=None):
        self.game_state    = game_state
        self.parent        = parent
        self.move          = move
        self.children      = []
        self.wins          = 0
        self.visits        = 0
        self.untried_moves = game_state.get_valid_moves()

    def add_child(self, move: Tuple[int, int]) -> 'MCTSNode':
        child_state = self.game_state.clone()
        child_state.make_move(move[0], move[1])
        child = MCTSNode(child_state, parent=self, move=move)
        self.untried_moves.remove(move)
        self.children.append(child)
        return child

    def select_child(self) -> 'MCTSNode':
        return max(self.children, key=lambda c: c.ucb_value())

    def ucb_value(self) -> float:
        if self.visits == 0:
            return float('inf')
        exploitation = self.wins / self.visits
        exploration = EXPLORATION_WEIGHT * math.sqrt(math.log(self.parent.visits) / self.visits)
        return exploitation + exploration

    def is_fully_expanded(self) -> bool:
        return len(self.untried_moves) == 0

    def is_terminal(self) -> bool:
        return self.game_state.game_over

def mcts_search(game_state: GameState, ai_player: int, sim_count: int) -> Tuple[int, int]:
    root = MCTSNode(game_state.clone())

    for _ in range(sim_count):
        node = root
        # Phase 1: Selection
        while node.is_fully_expanded() and not node.is_terminal():
            node = node.select_child()

        # Phase 2: Expansion
        if not node.is_terminal() and node.untried_moves:
            move = random.choice(node.untried_moves)
            node = node.add_child(move)

        # Phase 3: Simulation
        sim_state = node.game_state.clone()
        while not sim_state.game_over:
            valid = sim_state.get_valid_moves()
            if not valid:
                break
            mv = random.choice(valid)
            sim_state.make_move(mv[0], mv[1])

        # Phase 4: Backpropagation
        winner = sim_state.winner
        while node is not None:
            node.visits += 1
            if winner == ai_player:
                node.wins += 1
            node = node.parent

    # Choose the move from the most‐visited child
    if root.children:
        best = max(root.children, key=lambda c: c.visits)
        return best.move
    valid = game_state.get_valid_moves()
    return random.choice(valid) if valid else None
