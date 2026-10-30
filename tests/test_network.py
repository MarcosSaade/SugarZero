import numpy as np
import torch

import sugar
from network import (SugarNet, encode, action_index, mirror_planes, MIRROR_ACTION, MIRROR_SQ,
                     N_ACTIONS, N_PLANES, evaluate_batch)


def mirror_position(pos):
    code, mc, fwd = pos
    st = sugar.stacks(code)
    return sugar.from_stacks([st[MIRROR_SQ[i]] for i in range(9)], code >> 63, mc, fwd)


def random_positions(n, seed=0):
    rng = np.random.default_rng(seed)
    pos, out = sugar.START, []
    while len(out) < n:
        moves = sugar.legal_moves(pos)
        if not moves or pos[1] > 80:
            pos = sugar.START
            continue
        out.append(pos)
        pos = sugar.play(pos, moves[rng.integers(len(moves))])
    return out


def test_encode_is_canonical_for_side_to_move():
    # After a mirrored-and-colour-swapped reply, red sees exactly what blue saw at the start.
    x_blue = encode(sugar.START)
    red_start = (sugar.START[0] ^ sugar.TURN_BIT, 1, 0)
    assert encode(red_start)[:12].tolist() == x_blue[:12].tolist()
    assert x_blue.shape == (N_PLANES, 3, 3)


def test_action_indices_are_unique_and_in_range():
    for pos in random_positions(200):
        turn = pos[0] >> 63
        idx = [action_index(turn, m) for m in sugar.legal_moves(pos)]
        assert len(set(idx)) == len(idx)
        assert all(0 <= i < N_ACTIONS for i in idx)


def test_mirror_augmentation_matches_mirrored_positions():
    for pos in random_positions(200, seed=1):
        mpos = mirror_position(pos)
        assert np.array_equal(mirror_planes(encode(pos)), encode(mpos))
        turn = pos[0] >> 63
        for a, b in sugar.legal_moves(pos):
            mirrored_move = (MIRROR_SQ[a], MIRROR_SQ[b])
            assert mirrored_move in sugar.legal_moves(mpos)
            assert MIRROR_ACTION[action_index(turn, (a, b))] == action_index(turn, mirrored_move)


def test_evaluate_batch_outputs():
    torch.manual_seed(0)
    net = SugarNet(channels=16, blocks=1).eval()
    positions = random_positions(8, seed=2)
    priors, values = evaluate_batch(net, positions)
    for pos, pr in zip(positions, priors):
        assert len(pr) == len(sugar.legal_moves(pos))
        assert abs(pr.sum() - 1) < 1e-6
    assert values.shape == (8,) and np.all(np.abs(values) <= 1)
