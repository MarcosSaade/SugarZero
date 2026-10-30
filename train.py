# train.py
#
# AlphaZero self-play training for Sugar.
#   - workers play batches of games concurrently (batched MCTS), main process trains between rounds
#   - WDL value target from the true game result (draws by ply cap are kept, not discarded)
#   - left-right mirror augmentation, sliding-window replay buffer
#   - a fraction of games start from a short random opening for position diversity
#   - periodic evaluation against the exact solution (oracle is never used for training)

import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F

import sugar
from network import (SugarNet, encode, action_index, mirror_planes, MIRROR_ACTION, N_ACTIONS,
                    MAX_PLIES)
from mcts import search


def play_games(state_dict, net_cfg, n_games, sims, seed, cfg):
    torch.set_num_threads(1)
    rng = np.random.default_rng(seed)
    prng = random.Random(seed)
    net = SugarNet(**net_cfg)
    net.load_state_dict(state_dict)
    net.eval()

    games = []
    for _ in range(n_games):
        pos = sugar.START
        if prng.random() < cfg["random_start_frac"]:
            for _ in range(prng.randint(1, cfg["random_start_max"])):
                ms = sugar.legal_moves(pos)
                if not ms:
                    break
                pos = sugar.play(pos, prng.choice(ms))
            if not sugar.legal_moves(pos):
                pos = sugar.START
        games.append({"pos": pos, "hist": [], "z": None, "start_mc": pos[1]})

    X, PI, Z, stats = [], [], [], {"plies": 0, "results": [0, 0, 0]}
    live = list(range(n_games))
    while live:
        out = search(net, [games[i]["pos"] for i in live], sims, noise=True, rng=rng)
        still = []
        for i, (moves, visits, _) in zip(live, out):
            g = games[i]
            pos = g["pos"]
            turn = pos[0] >> 63
            pi = np.zeros(N_ACTIONS, dtype=np.float32)
            for m, n in zip(moves, visits):
                pi[action_index(turn, m)] = n
            pi /= pi.sum()
            g["hist"].append((encode(pos), pi, turn))
            if pos[1] - g["start_mc"] < cfg["temp_plies"]:
                p = visits ** (1.0 / cfg["temperature"])
                m = moves[rng.choice(len(moves), p=p / p.sum())]
            else:
                m = moves[int(np.argmax(visits))]
            pos = sugar.play(pos, m)
            g["pos"] = pos
            if not sugar.legal_moves(pos):
                g["winner"] = 1 - (pos[0] >> 63)
            elif pos[1] >= MAX_PLIES:
                g["winner"] = None
            else:
                still.append(i)
                continue
            # game over: label every position with WDL class from the mover's perspective
            for x, pi_, t in g["hist"]:
                if g["winner"] is None:
                    z = 1
                else:
                    z = 0 if g["winner"] == t else 2
                X.append(x); PI.append(pi_); Z.append(z)
            stats["plies"] += len(g["hist"])
            stats["results"][0 if g["winner"] is None else (1 if g["winner"] == 1 else 2)] += 1
        live = still
    return np.stack(X), np.stack(PI), np.array(Z, dtype=np.int64), stats


def train_steps(net, opt, buf, steps, batch_size, device):
    X, PI, Z = buf
    net.train()
    n = len(Z)
    tot_p = tot_v = 0.0
    for _ in range(steps):
        idx = np.random.randint(0, n, batch_size)
        x, pi, z = X[idx], PI[idx], Z[idx]
        flip = np.random.rand(batch_size) < 0.5
        x[flip] = mirror_planes(x[flip])
        pi[flip] = pi[flip][:, np.argsort(MIRROR_ACTION)]
        x = torch.from_numpy(x).to(device)
        pi = torch.from_numpy(pi).to(device)
        z = torch.from_numpy(z).to(device)
        logits, wdl = net(x)
        lp = -(pi * F.log_softmax(logits, 1)).sum(1).mean()
        lv = F.cross_entropy(wdl, z)
        opt.zero_grad()
        (lp + lv).backward()
        opt.step()
        tot_p += lp.item(); tot_v += lv.item()
    net.eval()
    return tot_p / steps, tot_v / steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/run1")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--games-per-worker", type=int, default=32)
    ap.add_argument("--sims", type=int, default=100)
    ap.add_argument("--buffer", type=int, default=400_000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--replay-ratio", type=float, default=4.0, help="sample reuse factor per new position")
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--channels", type=int, default=64)
    ap.add_argument("--blocks", type=int, default=4)
    ap.add_argument("--random-start-frac", type=float, default=0.25)
    ap.add_argument("--random-start-max", type=int, default=30)
    ap.add_argument("--temp-plies", type=int, default=12)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--eval-games", type=int, default=40)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    net_cfg = {"channels": args.channels, "blocks": args.blocks}
    net = SugarNet(**net_cfg).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    cfg = {"random_start_frac": args.random_start_frac, "random_start_max": args.random_start_max,
           "temp_plies": args.temp_plies, "temperature": args.temperature}
    log_path = os.path.join(args.out, "log.jsonl")
    start_it = 0
    if args.resume and os.path.exists(os.path.join(args.out, "latest.pt")):
        ck = torch.load(os.path.join(args.out, "latest.pt"), map_location=device)
        net.load_state_dict(ck["net"]); opt.load_state_dict(ck["opt"]); start_it = ck["iter"] + 1
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    oracle = None
    if args.eval_every:
        from oracle import Oracle
        from evaluate import eval_testset, play_vs_perfect
        oracle = Oracle()

    buf = None
    total_games = 0
    ctx = mp.get_context("spawn")
    with ctx.Pool(args.workers) as pool:
        for it in range(start_it, args.iters):
            t0 = time.time()
            sd = {k: v.cpu() for k, v in net.state_dict().items()}
            jobs = [(sd, net_cfg, args.games_per_worker, args.sims, args.seed * 100003 + it * 1000 + w, cfg)
                    for w in range(args.workers)]
            results = pool.starmap(play_games, jobs)
            X = np.concatenate([r[0] for r in results])
            PI = np.concatenate([r[1] for r in results])
            Z = np.concatenate([r[2] for r in results])
            res = np.sum([r[3]["results"] for r in results], axis=0)
            plies = sum(r[3]["plies"] for r in results)
            total_games += int(res.sum())
            if buf is None:
                buf = (X, PI, Z)
            else:
                buf = tuple(np.concatenate([b, n])[-args.buffer:] for b, n in zip(buf, (X, PI, Z)))
            t1 = time.time()
            steps = max(1, int(args.replay_ratio * len(Z) / args.batch))
            lp, lv = train_steps(net, opt, buf, steps, args.batch, device)
            t2 = time.time()
            rec = {"iter": it, "games": total_games, "positions": int(len(Z)), "buffer": int(len(buf[2])),
                   "draws": int(res[0]), "blue_wins": int(res[1]), "red_wins": int(res[2]),
                   "avg_len": plies / max(1, res.sum()), "loss_p": lp, "loss_v": lv,
                   "t_selfplay": t1 - t0, "t_train": t2 - t1}
            ck = {"net": net.state_dict(), "opt": opt.state_dict(), "iter": it, "net_cfg": net_cfg}
            torch.save(ck, os.path.join(args.out, "latest.pt"))
            if oracle is not None and (it + 1) % args.eval_every == 0:
                torch.save({"net": net.state_dict(), "net_cfg": net_cfg, "iter": it},
                           os.path.join(args.out, f"net_{it + 1:04d}.pt"))
                net.cpu()
                rec.update(eval_testset(net, sims=args.sims, limit=1500))
                pv = play_vs_perfect(net, oracle, games=args.eval_games, sims=args.sims * 2, seed=it)
                rec.update({f"vs_perfect_{k}": v for k, v in pv.items()})
                net.to(device)
                rec["t_eval"] = time.time() - t2
            print(json.dumps(rec), flush=True)
            with open(log_path, "a") as f:
                f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
