# sugar.py
#
# Fast integer-encoded Sugar engine (same encoding as solver/solve.c).
# A position is (code, mc, fwd):
#   code: 9 stacks x 7 bits + turn at bit 63. Stack code = (1 << h) | colour bits (bit i = piece at level i).
#   mc:   move count (only matters for the opening forward-move rule)
#   fwd:  bitmask, bit p set once player p has made a forward move
# Blue (1) starts at the bottom row and moves first; forward is "up" for blue and "down" for red.

from functools import lru_cache

ADJ = [[1, 3], [0, 2, 4], [1, 5], [0, 4, 6], [1, 3, 5, 7], [2, 4, 8], [3, 7], [4, 6, 8], [5, 7]]
MAX_H = 6
TURN_BIT = 1 << 63

H = [0] * 128
for _c in range(1, 128):
    H[_c] = _c.bit_length() - 1
TOP = [-1] + [(_c >> (H[_c] - 1)) & 1 if H[_c] else -1 for _c in range(1, 128)]
POP = [0] + [((_c & ((1 << (H[_c] - 1)) - 1)) | (1 << (H[_c] - 1))) if H[_c] else 0 for _c in range(1, 128)]
PUSH = [[0] * 128 for _ in range(2)]
for _col in range(2):
    for _c in range(1, 128):
        h = H[_c]
        if h < MAX_H:
            PUSH[_col][_c] = (_c ^ (1 << h)) | (_col << h) | (1 << (h + 1))

START_CODE = TURN_BIT
for _i, _s in enumerate([4, 4, 4, 1, 1, 1, 7, 7, 7]):
    START_CODE |= _s << (7 * _i)
START = (START_CODE, 0, 0)


def is_forward(turn, a, b):
    return (b // 3 > a // 3) if turn == 0 else (b // 3 < a // 3)


@lru_cache(maxsize=1_000_000)
def raw_moves(code):
    """Legal moves ignoring the opening rule, as a tuple of (a, b)."""
    turn = code >> 63
    out = []
    for a in range(9):
        ca = (code >> (7 * a)) & 127
        ha = H[ca]
        if not ha or TOP[ca] != turn:
            continue
        for b in ADJ[a]:
            hb = H[(code >> (7 * b)) & 127]
            if hb < MAX_H and ha >= hb:
                out.append((a, b))
    return tuple(out)


def player_moves_made(mc, turn):
    return mc // 2 if turn == 1 else (mc - 1) // 2


def rule_active(mc, fwd, turn):
    return not (fwd >> turn) & 1 and player_moves_made(mc, turn) < 3


def legal_moves(pos):
    code, mc, fwd = pos
    moves = raw_moves(code)
    turn = code >> 63
    if rule_active(mc, fwd, turn) and player_moves_made(mc, turn) == 2:
        moves = tuple(m for m in moves if is_forward(turn, *m))
    return moves


def rule_irrelevant(pos):
    """True once the opening rule can no longer restrict either player."""
    code, mc, fwd = pos
    turn = code >> 63
    for p in (0, 1):
        n = player_moves_made(mc if p == turn else mc + 1, p)
        if not (fwd >> p) & 1 and n < 3:
            return False
    return True


def play(pos, move):
    code, mc, fwd = pos
    a, b = move
    turn = code >> 63
    ca = (code >> (7 * a)) & 127
    cb = (code >> (7 * b)) & 127
    code &= ~((127 << (7 * a)) | (127 << (7 * b)))
    code |= (POP[ca] << (7 * a)) | (PUSH[turn][cb] << (7 * b))
    code ^= TURN_BIT
    if is_forward(turn, a, b):
        fwd |= 1 << turn
    return (code, mc + 1, fwd)


def from_stacks(stacks_, turn, mc=6, fwd=3):
    """Build a position from 9 bottom-up colour lists. Defaults put the opening rule out of play."""
    code = TURN_BIT if turn == 1 else 0
    for i, pieces in enumerate(stacks_):
        c = 1 << len(pieces)
        for lvl, p in enumerate(pieces):
            c |= p << lvl
        code |= c << (7 * i)
    return (code, mc, fwd)


def stacks(code):
    """List of 9 stacks, each a list of colours bottom-up."""
    out = []
    for i in range(9):
        c = (code >> (7 * i)) & 127
        out.append([(c >> lvl) & 1 for lvl in range(H[c])])
    return out
