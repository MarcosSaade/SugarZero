# oracle.py
#
# Perfect-play oracle backed by the exact retrograde solution from solver/solve.c.
# Used ONLY for evaluation (never as a training signal), so the agent stays tabula rasa.
#
# value: +1 side to move wins, 0 draw, -1 side to move loses (with optimal play)
# depth: plies until the game ends under optimal play (winner fastest, loser slowest)

import os
import random
import numpy as np

import sugar

SOLUTION_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "solver", "solution.bin")


class Oracle:
    def __init__(self, path=SOLUTION_PATH):
        with open(path, "rb") as f:
            n = int(np.frombuffer(f.read(8), dtype=np.uint64)[0])
            keys = np.frombuffer(f.read(8 * n), dtype=np.uint64)
            vals = np.frombuffer(f.read(n), dtype=np.int8)
            deps = np.frombuffer(f.read(2 * n), dtype=np.uint16)
        order = np.argsort(keys)
        self.keys = keys[order]
        self.vals = vals[order]
        self.deps = deps[order]
        self._opening = {}

    def lookup_codes(self, codes):
        """Vectorised (value, depth) for raw codes; ignores the opening rule."""
        q = np.asarray(codes, dtype=np.uint64)
        idx = np.searchsorted(self.keys, q)
        idx = np.minimum(idx, len(self.keys) - 1)
        if not np.all(self.keys[idx] == q):
            raise KeyError("position not in solved table")
        return self.vals[idx].astype(np.int64), self.deps[idx].astype(np.int64)

    def value(self, pos):
        """(value, depth) for a full position, respecting the opening rule."""
        if sugar.rule_irrelevant(pos):
            v, d = self.lookup_codes([pos[0]])
            return int(v[0]), int(d[0])
        if pos in self._opening:
            return self._opening[pos]
        moves = sugar.legal_moves(pos)
        if not moves:
            res = (-1, 0)
        else:
            res = combine([self.value(sugar.play(pos, m)) for m in moves])
        self._opening[pos] = res
        return res

    def move_values(self, pos):
        """{move: (value for the mover, depth)} for every legal move."""
        out = {}
        for m in sugar.legal_moves(pos):
            v, d = self.value(sugar.play(pos, m))
            out[m] = (-v, d + 1)
        return out

    def perfect_move(self, pos, rng=random):
        """Optimal move: fastest win, slowest loss, random among draw-holding moves."""
        mv = self.move_values(pos)
        best = max(v for v, _ in mv.values())
        cands = [(m, d) for m, (v, d) in mv.items() if v == best]
        if best == 1:
            dmin = min(d for _, d in cands)
            cands = [c for c in cands if c[1] == dmin]
        elif best == -1:
            dmax = max(d for _, d in cands)
            cands = [c for c in cands if c[1] == dmax]
        return rng.choice(cands)[0]


def combine(child_results):
    """Negamax over children's (value, depth) from the opponent's perspective."""
    losses = [d for v, d in child_results if v == -1]
    if losses:
        return (1, 1 + min(losses))
    if any(v == 0 for v, _ in child_results):
        return (0, 0)
    return (-1, 1 + max(d for _, d in child_results))
