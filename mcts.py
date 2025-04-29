# mcts.py

import math
import random
import torch

from constants import EXPLORATION_WEIGHT, DIRICHLET_ALPHA, NOISE_EPSILON
from game_state import GameState
from utils import encode_board, move_to_index, dirichlet_noise

# Number of leaf evaluations to batch together
EVAL_BATCH_SIZE = 16

class PUCTNode:
    def __init__(
        self,
        game_state: GameState,
        parent=None,
        move=None,
        prior: float = 1.0,
        policy_model=None,
        device: str = 'cpu'
    ):
        self.game_state    = game_state
        self.parent        = parent
        self.move          = move
        self.prior         = prior
        self.W             = 0.0
        self.N             = 0
        self.children      = {}               # move → PUCTNode
        self.untried_moves = game_state.get_valid_moves()
        self.policy_model  = policy_model
        self.device        = device
        self._init_priors()

    def _init_priors(self):
        # if no policy guidance, leave empty
        if not self.untried_moves or self.policy_model is None:
            self.priors = {}
            return

        tensor, turn = encode_board(self.game_state.board, self.game_state.turn)
        b = torch.from_numpy(tensor).unsqueeze(0).to(self.device)
        t = torch.tensor([turn], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            probs = self.policy_model(b, t).squeeze(0).cpu().numpy()

        self.priors = {
            move: float(probs[move_to_index(move)])
            for move in self.untried_moves
        }

    def select_child(self, cpuct: float):
        total_N = sum(child.N for child in self.children.values()) or 1

        def score(child: 'PUCTNode'):
            Q = child.W / child.N if child.N > 0 else 0.0
            U = cpuct * child.prior * math.sqrt(total_N) / (1 + child.N)
            return Q + U

        return max(self.children.values(), key=score)

    def expand(self, move: tuple[int, int]) -> 'PUCTNode':
        prior = self.priors.get(move, 0.0)
        child_state = self.game_state.clone()
        child_state.make_move(*move)
        child = PUCTNode(
            child_state,
            parent=self,
            move=move,
            prior=prior,
            policy_model=self.policy_model,
            device=self.device
        )
        self.untried_moves.remove(move)
        self.children[move] = child
        return child

def _evaluate_and_backprop(nodes, root_player, value_model, device):
    tensors, turns = [], []
    for node in nodes:
        tensor, turn = encode_board(node.game_state.board, node.game_state.turn)
        tensors.append(torch.from_numpy(tensor).unsqueeze(0))
        turns.append(turn)

    batch_states = torch.cat(tensors).to(device)
    batch_turns  = torch.tensor(turns, dtype=torch.float32, device=device)
    with torch.no_grad():
        v_batch = value_model(batch_states, batch_turns).squeeze(1).cpu().numpy()

    for node, v in zip(nodes, v_batch):
        cur = node
        while cur is not None:
            cur.N += 1
            if cur.game_state.turn == root_player:
                cur.W += v
            else:
                cur.W += (1 - v)
            cur = cur.parent

def puct_search(
    game_state: GameState,
    sim_count: int,
    policy_model,
    value_model,
    cpuct: float = EXPLORATION_WEIGHT,
    device: str = 'cpu'
) -> tuple[int, int]:

    # Root node
    root = PUCTNode(game_state.clone(), policy_model=policy_model, device=device)
    # Inject Dirichlet noise at root priors
    if root.priors:
        root.priors = dirichlet_noise(root.priors, DIRICHLET_ALPHA, NOISE_EPSILON)

    root_player = game_state.turn
    eval_batch   = []

    for _ in range(sim_count):
        node = root

        # SELECTION
        while not node.untried_moves and node.children:
            node = node.select_child(cpuct)

        # EXPANSION
        if node.untried_moves:
            mv = random.choice(node.untried_moves)
            node = node.expand(mv)

        # COLLECT for batch eval
        eval_batch.append(node)
        if len(eval_batch) >= EVAL_BATCH_SIZE:
            _evaluate_and_backprop(eval_batch, root_player, value_model, device)
            eval_batch.clear()

    # final batch
    if eval_batch:
        _evaluate_and_backprop(eval_batch, root_player, value_model, device)

    # pick most‐visited child
    if root.children:
        best_move = max(root.children.items(), key=lambda item: item[1].N)[0]
    else:
        valid = game_state.get_valid_moves()
        best_move = random.choice(valid) if valid else None

    return best_move

def puct_search_with_policy(
    game_state: GameState,
    sim_count: int,
    policy_model,
    value_model,
    cpuct: float = EXPLORATION_WEIGHT,
    device: str = 'cpu'
) -> tuple[tuple[int, int] | None, dict[tuple[int, int], float]]:

    root = PUCTNode(game_state.clone(), policy_model=policy_model, device=device)
    if root.priors:
        root.priors = dirichlet_noise(root.priors, DIRICHLET_ALPHA, NOISE_EPSILON)

    root_player = game_state.turn
    eval_batch   = []

    for _ in range(sim_count):
        node = root
        while not node.untried_moves and node.children:
            node = node.select_child(cpuct)
        if node.untried_moves:
            mv = random.choice(node.untried_moves)
            node = node.expand(mv)
        eval_batch.append(node)
        if len(eval_batch) >= EVAL_BATCH_SIZE:
            _evaluate_and_backprop(eval_batch, root_player, value_model, device)
            eval_batch.clear()

    if eval_batch:
        _evaluate_and_backprop(eval_batch, root_player, value_model, device)

    if root.children:
        visit_counts = {m: c.N for m, c in root.children.items()}
        total = sum(visit_counts.values()) or 1
        policy_dist = {m: cnt/total for m, cnt in visit_counts.items()}
        best_move = max(visit_counts.items(), key=lambda item: item[1])[0]
    else:
        valid = game_state.get_valid_moves()
        best_move   = random.choice(valid) if valid else None
        policy_dist = {m: 1/len(valid) for m in valid} if valid else {}

    return best_move, policy_dist
