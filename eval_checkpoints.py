# eval_checkpoints.py

import os
import re
import argparse
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
import matplotlib.pyplot as plt

from mcts import puct_search_with_policy
from policy_net import PolicyNet
from value_net import ValueNet
from game_state import GameState
from constants import EXPLORATION_WEIGHT, MIN_MCTS_SIMULATIONS

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model(checkpoint_dir, idx):
    policy = PolicyNet().to(DEVICE)
    value  = ValueNet().to(DEVICE)
    ppath = os.path.join(checkpoint_dir, f'policy_model_{idx}.pt')
    vpath = os.path.join(checkpoint_dir, f'value_model_{idx}.pt')
    policy.load_state_dict(torch.load(ppath, map_location=DEVICE))
    value.load_state_dict(torch.load(vpath, map_location=DEVICE))
    policy.eval()
    value.eval()
    return policy, value

def play_one_game(model_a, value_a, model_b, value_b, start_player):
    """
    model_a/value_a when game.turn == start_player,
    model_b/value_b otherwise.
    Returns: 1 if A wins, -1 if B wins, 0 if draw.
    """
    game = GameState()
    while not game.game_over and game.move_count < 200:
        if game.turn == start_player:
            move, _ = puct_search_with_policy(
                game,
                sim_count=MIN_MCTS_SIMULATIONS,
                policy_model=model_a,
                value_model=value_a,
                cpuct=EXPLORATION_WEIGHT,
                device=DEVICE
            )
        else:
            move, _ = puct_search_with_policy(
                game,
                sim_count=MIN_MCTS_SIMULATIONS,
                policy_model=model_b,
                value_model=value_b,
                cpuct=EXPLORATION_WEIGHT,
                device=DEVICE
            )
        game.make_move(*move)

    if game.winner is None:
        return 0
    return 1 if game.winner == start_player else -1

def evaluate_pair(policy_a, value_a, policy_b, value_b, games, workers):
    wins = losses = draws = 0
    with ThreadPoolExecutor(max_workers=workers) as exec:
        futures = [
            exec.submit(play_one_game, policy_a, value_a, policy_b, value_b, i % 2)
            for i in range(games)
        ]
        for f in as_completed(futures):
            res = f.result()
            if res == 1:
                wins += 1
            elif res == -1:
                losses += 1
            else:
                draws += 1
    return wins, losses, draws

def main():
    parser = argparse.ArgumentParser("Evaluate latest model vs previous checkpoints")
    parser.add_argument("--dir", type=str, required=True, help="Checkpoint directory")
    parser.add_argument("--games", type=int, default=25, help="Games per pair")
    parser.add_argument("--workers", type=int, default=os.cpu_count(), help="Parallel workers")
    args = parser.parse_args()

    files = os.listdir(args.dir)
    pattern = re.compile(r'policy_model_(\d+)\.pt')
    idxs = sorted(int(m.group(1)) for f in files if (m := pattern.match(f)))
    if len(idxs) < 2:
        print("Need at least two checkpoints.")
        return

    latest = idxs[-1]
    print(f"Latest checkpoint: {latest}")
    policy_new, value_new = load_model(args.dir, latest)

    results = []
    for prev in idxs[:-1]:
        print(f"\nEvaluating {latest} vs {prev} over {args.games} games…")
        policy_old, value_old = load_model(args.dir, prev)
        w, l, d = evaluate_pair(
            policy_new, value_new, policy_old, value_old, args.games, args.workers
        )
        total = w + l + d
        print(f"  Wins:   {w}/{total} ({w/total:.2%})")
        print(f"  Losses: {l}/{total} ({l/total:.2%})")
        print(f"  Draws:  {d}/{total} ({d/total:.2%})")
        results.append((prev, w/total*100, l/total*100, d/total*100))

    # Plot
    ckpts, win_p, loss_p, draw_p = zip(*results)
    plt.figure()
    plt.plot(ckpts, win_p, marker='o', label='Win %')
    plt.plot(ckpts, loss_p, marker='o', label='Loss %')
    plt.plot(ckpts, draw_p, marker='o', label='Draw %')
    plt.xlabel('Opponent Checkpoint')
    plt.ylabel('Percentage')
    plt.title(f'Model {latest} vs Previous Checkpoints')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # save results to file
    with open(os.path.join(args.dir, f"eval_{latest}.txt"), "w") as f:
        for prev, w, l, d in results:
            f.write(f"{prev}: {w:.2f}% wins, {l:.2f}% losses, {d:.2f}% draws\n")
    print(f"Results saved to {os.path.join(args.dir, f'eval_{latest}.txt')}")
    print("Evaluation complete.")

    # save the plot
    plt.savefig(os.path.join(args.dir, f"eval_{latest}.png"))
    print(f"Plot saved to {os.path.join(args.dir, f'eval_{latest}.png')}")
    print("Plot saved.")

if __name__ == "__main__":
    main()
