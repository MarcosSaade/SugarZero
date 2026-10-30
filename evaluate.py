# evaluate.py
#
# Ground-truth evaluation against the exact solution (oracle is never used for training).
#   1. Test set of oracle-labelled positions -> value-head WDL accuracy, raw-policy and MCTS
#      "optimal move" rate on positions where the choice matters.
#   2. Matches from the start position against a perfect player (who never loses from a draw).

import os
import random
import numpy as np
import torch

import sugar
from network import SugarNet, encode, MAX_PLIES, evaluate_batch
from mcts import search

TESTSET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "solver", "testset.npz")


def _random_game_positions(rng, oracle, eps):
    """Positions from games where each move is perfect w.p. 1-eps, random otherwise."""
    pos, out = sugar.START, []
    while sugar.legal_moves(pos) and pos[1] < MAX_PLIES:
        out.append(pos)
        if rng.random() < eps:
            pos = sugar.play(pos, rng.choice(sugar.legal_moves(pos)))
        else:
            pos = sugar.play(pos, oracle.perfect_move(pos, rng))
    return out


def build_testset(oracle, n=4000, seed=1234):
    """
    Oracle-labelled test set. Move-choice metrics use only critical positions (where some moves
    change the game-theoretic result); value accuracy uses all sampled positions.
    """
    rng = random.Random(seed)
    pos_list, good = [], []
    vpos, vvals = [], []  # unfiltered positions for value accuracy (includes lost positions)
    while len(pos_list) < n:
        eps = rng.choice([0.1, 0.3, 0.6, 1.0])
        cands = _random_game_positions(rng, oracle, eps)
        rng.shuffle(cands)
        for p in cands[:3]:
            vpos.append(p)
            vvals.append(oracle.value(p)[0])
            mv = oracle.move_values(p)
            best = max(v for v, _ in mv.values())
            if all(v == best for v, _ in mv.values()):
                continue  # not a real decision
            pos_list.append(p)
            good.append([m for m, (v, _) in mv.items() if v == best])
    codes = np.array([p[0] for p in pos_list], dtype=np.uint64)
    mcs = np.array([p[1] for p in pos_list], dtype=np.int64)
    fwds = np.array([p[2] for p in pos_list], dtype=np.int64)
    good_flat = np.array([[i, a, b] for i, g in enumerate(good) for a, b in g], dtype=np.int64)
    np.savez(TESTSET_PATH, codes=codes, mcs=mcs, fwds=fwds, good=good_flat,
             vcodes=np.array([p[0] for p in vpos], dtype=np.uint64), vmcs=np.array([p[1] for p in vpos]),
             vfwds=np.array([p[2] for p in vpos]), vvalues=np.array(vvals))


def load_testset():
    d = np.load(TESTSET_PATH)
    positions = [(int(c), int(m), int(f)) for c, m, f in zip(d["codes"], d["mcs"], d["fwds"])]
    good = [set() for _ in positions]
    for i, a, b in d["good"]:
        good[i].add((int(a), int(b)))
    vpositions = [(int(c), int(m), int(f)) for c, m, f in zip(d["vcodes"], d["vmcs"], d["vfwds"])]
    return positions, good, vpositions, d["vvalues"]


@torch.no_grad()
def eval_testset(net, sims=100, limit=None):
    positions, good, vpositions, values = load_testset()
    if limit:
        positions, good = positions[:limit], good[:limit]
    net.eval()
    x = torch.from_numpy(np.stack([encode(p) for p in vpositions]))
    _, wdl = net(x)
    pred = np.array([1, 0, -1])[wdl.argmax(1).numpy()]
    value_acc = float((pred == values).mean())
    priors, _ = evaluate_batch(net, positions)
    pol_ok = np.mean([sugar.legal_moves(p)[int(np.argmax(pr))] in g for p, pr, g in zip(positions, priors, good)])
    res = {"value_acc": value_acc, "policy_opt": float(pol_ok)}
    if sims:
        out = []
        for i in range(0, len(positions), 256):
            out += search(net, positions[i:i + 256], sims)
        res["mcts_opt"] = float(np.mean([m[int(np.argmax(n))] in g for (m, n, _), g in zip(out, good)]))
    return res


def play_vs_perfect(net, oracle, games=40, sims=200, seed=0):
    """Agent alternates colours vs a perfect player. Returns dict of W/D/L and agent blunder rate."""
    rng = random.Random(seed)
    sides = [i % 2 for i in range(games)]
    pos = [sugar.START] * games
    result = [None] * games
    blunders = moves = 0
    while any(r is None for r in result):
        live = [i for i in range(games) if result[i] is None]
        agent_idx = [i for i in live if (pos[i][0] >> 63) == sides[i]]
        if agent_idx:
            out = search(net, [pos[i] for i in agent_idx], sims)
            for i, (mvs, n, _) in zip(agent_idx, out):
                m = mvs[int(np.argmax(n))]
                mv = oracle.move_values(pos[i])
                blunders += mv[m][0] < max(v for v, _ in mv.values())
                moves += 1
                pos[i] = sugar.play(pos[i], m)
        for i in live:
            if i not in agent_idx:
                pos[i] = sugar.play(pos[i], oracle.perfect_move(pos[i], rng))
        for i in live:
            if not sugar.legal_moves(pos[i]):
                result[i] = 1 if (pos[i][0] >> 63) != sides[i] else -1
            elif pos[i][1] >= MAX_PLIES:
                result[i] = 0
    return {"W": result.count(1), "D": result.count(0), "L": result.count(-1),
            "blunder_rate": blunders / max(moves, 1)}


if __name__ == "__main__":
    import argparse
    from oracle import Oracle
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--ckpt", type=str)
    ap.add_argument("--sims", type=int, default=200)
    ap.add_argument("--games", type=int, default=40)
    args = ap.parse_args()
    torch.set_num_threads(4)
    oracle = Oracle()
    if args.build:
        build_testset(oracle)
    if args.ckpt:
        net = SugarNet()
        net.load_state_dict(torch.load(args.ckpt, map_location="cpu")["net"])
        net.eval()
        print(eval_testset(net, sims=args.sims))
        print(play_vs_perfect(net, oracle, games=args.games, sims=args.sims))
