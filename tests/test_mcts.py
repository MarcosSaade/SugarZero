import numpy as np
import torch

import sugar
from mcts import search
from network import SugarNet

B, R = 1, 0


def untrained_net():
    torch.manual_seed(0)
    return SugarNet(channels=16, blocks=1).eval()


def test_finds_immediate_win():
    # Blue covers red's only piece (4 -> 1), leaving red without a move. Every other move lets red play.
    stacks = [[] for _ in range(9)]
    stacks[1] = [R]
    stacks[4] = [B, B]
    stacks[6] = [B]
    pos = sugar.from_stacks(stacks, B)
    moves, visits, value = search(untrained_net(), [pos], 200)[0]
    assert moves[int(np.argmax(visits))] == (4, 1)
    assert value > 0.5


def test_visit_counts_sum_to_simulations():
    (moves, visits, _), = search(untrained_net(), [sugar.START], 64)
    assert visits.sum() == 64
    assert len(moves) == len(visits) == len(sugar.legal_moves(sugar.START))


def test_batched_roots_are_independent():
    net = untrained_net()
    pos2 = sugar.play(sugar.START, (7, 4))
    a = search(net, [sugar.START], 50)[0]
    b = search(net, [sugar.START, pos2], 50)[0]
    assert a[0] == b[0] and np.array_equal(a[1], b[1])
