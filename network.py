# network.py
#
# Canonical-perspective encoding + residual policy/value network.
# The board is always seen from the side to move: if red (0) is to move, rows are mirrored and
# colours swapped, so "forward" is always up. Policy = 9 from-squares x 4 directions (canonical).

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import sugar

N_PLANES = 17
N_ACTIONS = 36
MAX_PLIES = 200

DIRS = {-3: 0, 3: 1, -1: 2, 1: 3}  # up, down, left, right (canonical frame)
FLIP_SQ = [6, 7, 8, 3, 4, 5, 0, 1, 2]
MIRROR_SQ = [2, 1, 0, 5, 4, 3, 8, 7, 6]

# STACK_PLANES[col][code] -> (12,) floats: levels 0..5 of "mine" then "theirs"
STACK_PLANES = np.zeros((2, 128, 12), dtype=np.float32)
for _me in range(2):
    for _c in range(1, 128):
        for _lvl in range(sugar.H[_c]):
            _p = (_c >> _lvl) & 1
            STACK_PLANES[_me, _c, _lvl + (0 if _p == _me else 6)] = 1.0

# Mirror (left-right) permutation of action indices, used for data augmentation.
MIRROR_ACTION = np.zeros(N_ACTIONS, dtype=np.int64)
for _a in range(9):
    for _delta, _d in DIRS.items():
        _md = {2: 3, 3: 2}.get(_d, _d)
        MIRROR_ACTION[_a * 4 + _d] = MIRROR_SQ[_a] * 4 + _md


def action_index(turn, move):
    a, b = move
    if turn == 0:
        a, b = FLIP_SQ[a], FLIP_SQ[b]
    return a * 4 + DIRS[b - a]


def encode(pos):
    code, mc, fwd = pos
    turn = code >> 63
    x = np.zeros((N_PLANES, 9), dtype=np.float32)
    for i in range(9):
        j = FLIP_SQ[i] if turn == 0 else i
        x[:12, j] = STACK_PLANES[turn, (code >> (7 * i)) & 127]
    opp = 1 - turn
    x[12] = (fwd >> turn) & 1
    x[13] = (fwd >> opp) & 1
    x[14] = min(sugar.player_moves_made(mc, turn), 3) / 3
    x[15] = min(sugar.player_moves_made(mc + 1, opp), 3) / 3
    x[16] = mc / MAX_PLIES
    return x.reshape(N_PLANES, 3, 3)


def mirror_planes(x):
    return x[..., ::-1].copy()


class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.b1 = nn.BatchNorm2d(ch)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.b2 = nn.BatchNorm2d(ch)

    def forward(self, x):
        y = F.relu(self.b1(self.c1(x)))
        y = self.b2(self.c2(y))
        return F.relu(x + y)


class SugarNet(nn.Module):
    def __init__(self, channels=64, blocks=4):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(N_PLANES, channels, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(channels), nn.ReLU())
        self.body = nn.Sequential(*[ResBlock(channels) for _ in range(blocks)])
        self.p_conv = nn.Sequential(nn.Conv2d(channels, 8, 1, bias=False), nn.BatchNorm2d(8), nn.ReLU())
        self.p_fc = nn.Linear(8 * 9, N_ACTIONS)
        self.v_conv = nn.Sequential(nn.Conv2d(channels, 4, 1, bias=False), nn.BatchNorm2d(4), nn.ReLU())
        self.v_fc = nn.Sequential(nn.Linear(4 * 9, 64), nn.ReLU(), nn.Linear(64, 3))

    def forward(self, x):
        """Returns policy logits (B,36) and WDL logits (B,3) ordered [win, draw, loss]."""
        h = self.body(self.stem(x))
        p = self.p_fc(self.p_conv(h).flatten(1))
        v = self.v_fc(self.v_conv(h).flatten(1))
        return p, v


@torch.no_grad()
def evaluate_batch(net, positions):
    """-> (list of {move: prior}, np.array of scalar values in [-1, 1] for the side to move)."""
    x = torch.from_numpy(np.stack([encode(p) for p in positions]))
    logits, wdl = net(x)
    probs = wdl.softmax(1).numpy()
    values = probs[:, 0] - probs[:, 2]
    logits = logits.numpy()
    priors = []
    for i, pos in enumerate(positions):
        moves = sugar.legal_moves(pos)
        turn = pos[0] >> 63
        lg = np.array([logits[i, action_index(turn, m)] for m in moves], dtype=np.float64)
        lg = np.exp(lg - lg.max())
        lg /= lg.sum()
        priors.append(lg)
    return priors, values
