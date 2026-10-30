import os
import random

import pytest

import sugar
from oracle import SOLUTION_PATH, combine

pytestmark = pytest.mark.skipif(not os.path.exists(SOLUTION_PATH),
                                reason="run solver/solve first to generate the solution")


@pytest.fixture(scope="module")
def oracle():
    from oracle import Oracle
    return Oracle()


def test_start_is_a_draw_and_forward_openings_lose(oracle):
    assert oracle.value(sugar.START) == (0, 0)
    values = oracle.move_values(sugar.START)
    for move, (v, _) in values.items():
        is_forward = move[1] // 3 < move[0] // 3
        assert v == (-1 if is_forward else 0)


def test_values_satisfy_negamax(oracle):
    rng = random.Random(0)
    pos = sugar.START
    for _ in range(300):
        moves = sugar.legal_moves(pos)
        if not moves or pos[1] > 150:
            pos = sugar.START
            continue
        children = [oracle.value(sugar.play(pos, m)) for m in moves]
        assert combine(children)[0] == oracle.value(pos)[0]
        pos = sugar.play(pos, rng.choice(moves))
