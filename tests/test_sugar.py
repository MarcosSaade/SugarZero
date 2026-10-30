import sugar

B, R = 1, 0
EMPTY = []


def test_start_position():
    assert sugar.stacks(sugar.START[0]) == [[R, R]] * 3 + [EMPTY] * 3 + [[B, B]] * 3
    assert sugar.START[0] >> 63 == B
    assert sorted(sugar.legal_moves(sugar.START)) == [(6, 3), (6, 7), (7, 4), (7, 6), (7, 8), (8, 5), (8, 7)]


def test_play_moves_top_piece_and_switches_turn():
    pos = sugar.play(sugar.START, (7, 4))
    st = sugar.stacks(pos[0])
    assert st[7] == [B] and st[4] == [B]
    assert pos[0] >> 63 == R and pos[1] == 1 and pos[2] == 2  # blue has moved forward


def test_cannot_move_onto_taller_stack():
    stacks = [[] for _ in range(9)]
    stacks[4] = [B]
    stacks[1] = [R, R]
    stacks[3] = [R]
    pos = sugar.from_stacks(stacks, B)
    moves = sugar.legal_moves(pos)
    assert (4, 1) not in moves     # height 1 onto height 2
    assert (4, 3) in moves         # equal heights are allowed
    assert (4, 5) in moves and (4, 7) in moves


def test_stack_capacity_is_six():
    stacks = [[] for _ in range(9)]
    stacks[0] = [R] * 6
    stacks[1] = [B] * 6
    pos = sugar.from_stacks(stacks, B)
    assert (1, 0) not in sugar.legal_moves(pos)
    assert (1, 2) in sugar.legal_moves(pos)


def test_player_without_moves_has_lost():
    stacks = [[] for _ in range(9)]
    stacks[0] = [B, B, B, R]  # every blue piece is buried under red
    stacks[2] = [B, B, B, R]
    pos = sugar.from_stacks(stacks, B)
    assert sugar.legal_moves(pos) == ()


def test_forward_rule_forces_third_move_for_both_players():
    # Both players shuffle sideways twice; on their third move only forward moves remain.
    pos = sugar.START
    for move in [(6, 7), (0, 1), (7, 6), (1, 0)]:
        assert move in sugar.legal_moves(pos)
        pos = sugar.play(pos, move)
    assert all(b // 3 < a // 3 for a, b in sugar.legal_moves(pos))  # blue: only upward moves
    pos = sugar.play(pos, (6, 3))
    assert all(b // 3 > a // 3 for a, b in sugar.legal_moves(pos))  # red: only downward moves
    pos = sugar.play(pos, (0, 3))
    assert sugar.rule_irrelevant(pos)
    assert any(b // 3 == a // 3 for a, b in sugar.legal_moves(pos))


def test_rule_irrelevant_once_both_moved_forward():
    pos = sugar.play(sugar.START, (7, 4))
    assert not sugar.rule_irrelevant(pos)
    pos = sugar.play(pos, (1, 4))
    assert sugar.rule_irrelevant(pos)
