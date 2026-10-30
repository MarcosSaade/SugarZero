# mcts.py
#
# AlphaZero PUCT search, run on many independent roots at once so leaf evaluations are batched.
# Values are always from the perspective of the side to move at a node (negamax backup).
# Terminal positions use the true result (no legal moves = loss, ply cap = draw), never the net.

import math
import numpy as np

import sugar
from network import MAX_PLIES, evaluate_batch

C_PUCT = 1.5
FPU_REDUCTION = 0.25
DIRICHLET_ALPHA = 0.5
NOISE_EPS = 0.25


def terminal_value(pos):
    if not sugar.legal_moves(pos):
        return -1.0
    if pos[1] >= MAX_PLIES:
        return 0.0
    return None


class Node:
    __slots__ = ("pos", "moves", "P", "N", "W", "children", "terminal", "n_total", "v_sum")

    def __init__(self, pos):
        self.pos = pos
        self.terminal = terminal_value(pos)
        self.moves = None
        self.P = None
        self.N = None
        self.W = None
        self.children = None
        self.n_total = 0
        self.v_sum = 0.0

    def expand(self, priors):
        self.moves = sugar.legal_moves(self.pos)
        self.P = list(priors)
        k = len(self.moves)
        self.N = [0] * k
        self.W = [0.0] * k
        self.children = [None] * k

    def select(self):
        sqrt_n = math.sqrt(self.n_total + 1)
        # first-play urgency: unvisited children assumed a bit worse than the parent's mean value
        seen = sum(p for p, n in zip(self.P, self.N) if n)
        parent_q = self.v_sum / self.n_total if self.n_total else 0.0
        fpu = parent_q - FPU_REDUCTION * math.sqrt(seen)
        best, best_i = -1e9, 0
        for i in range(len(self.moves)):
            n = self.N[i]
            q = self.W[i] / n if n else fpu
            s = q + C_PUCT * self.P[i] * sqrt_n / (1 + n)
            if s > best:
                best, best_i = s, i
        return best_i


def _backup(path, v):
    # path: list of (node, child_idx); v: value for the side to move at the leaf
    for node, i in reversed(path):
        v = -v
        node.N[i] += 1
        node.W[i] += v
        node.n_total += 1
        node.v_sum += v


def _add_noise(node, rng):
    if len(node.moves) > 1:
        eta = rng.dirichlet([DIRICHLET_ALPHA] * len(node.moves))
        node.P = [(1 - NOISE_EPS) * p + NOISE_EPS * e for p, e in zip(node.P, eta)]


def search(net, positions, sims, noise=False, rng=None):
    """
    Run `sims` simulations from each root position (all non-terminal).
    Returns list of (moves, visit_counts, root_value) per root.
    """
    rng = rng or np.random.default_rng()
    roots = [Node(p) for p in positions]
    priors, _ = evaluate_batch(net, positions)
    for r, pr in zip(roots, priors):
        r.expand(pr)
        if noise:
            _add_noise(r, rng)

    for _ in range(sims):
        pending = []
        for root in roots:
            node, path = root, []
            while True:
                i = node.select()
                path.append((node, i))
                child = node.children[i]
                if child is None:
                    child = Node(sugar.play(node.pos, node.moves[i]))
                    node.children[i] = child
                    break
                if child.terminal is not None or child.moves is None:
                    break
                node = child
            if child.terminal is not None:
                _backup(path, child.terminal)
            else:
                pending.append((child, path))
        if pending:
            pr, vals = evaluate_batch(net, [c.pos for c, _ in pending])
            for (child, path), p, v in zip(pending, pr, vals):
                if child.moves is None:  # may have been reached twice this round
                    child.expand(p)
                _backup(path, float(v))

    out = []
    for r in roots:
        q = r.v_sum / r.n_total if r.n_total else 0.0
        out.append((r.moves, np.array(r.N, dtype=np.float64), q))
    return out
