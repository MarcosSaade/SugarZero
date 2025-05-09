# eval_checkpoints.py

import os
import re
import argparse
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
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
    return policy, value

def _dummy_value(states, turns):
    """Picklable dummy value model always returning 0.5."""
    bsz = states.size(0)
    return torch.full((bsz, 1), 0.5, device=DEVICE)

def play_one_game(model_a, model_b, start_player):
    """
    model_a plays when game.turn == start_player,
    model_b otherwise.
    Returns:
        1 if model_a wins,
       -1 if model_b wins,
        0 if draw.
    """
    game = GameState()
    while not game.game_over and game.move_count < 200:
        if game.turn == start_player:
            move, _ = puct_search_with_policy(
                game,
                sim_count=MIN_MCTS_SIMULATIONS,
                policy_model=model_a,
                value_model=_dummy_value,
                cpuct=EXPLORATION_WEIGHT,
                device=DEVICE
            )
        else:
            move, _ = puct_search_with_policy(
                game,
                sim_count=MIN_MCTS_SIMULATIONS,
                policy_model=model_b,
                value_model=_dummy_value,
                cpuct=EXPLORATION_WEIGHT,
                device=DEVICE
            )
        game.make_move(*move)

    # determine outcome
    if game.winner is None:
        return 0
    elif game.winner == start_player:
        return 1
    else:
        return -1

def evaluate_pair(model_a, model_b, games, workers):
    wins = 0
    losses = 0
    draws = 0

    with ThreadPoolExecutor(max_workers=workers) as exec:
        futures = [exec.submit(play_one_game, model_a, model_b, i % 2)
                   for i in range(games)]
        for f in as_completed(futures):
            result = f.result()
            if result == 1:
                wins += 1
            elif result == -1:
                losses += 1
            else:
                draws += 1

    return wins, losses, draws

def main():
    parser = argparse.ArgumentParser(description="Evaluate latest model vs previous checkpoints")
    parser.add_argument("--dir",   type=str, required=True,
                        help="Checkpoint directory")
    parser.add_argument("--games", type=int, default=25,
                        help="Number of games per pair")
    parser.add_argument("--workers", type=int, default=os.cpu_count(),
                        help="Parallel workers")
    args = parser.parse_args()

    # find all checkpoint indices
    files = os.listdir(args.dir)
    pattern = re.compile(r'policy_model_(\d+)\.pt')
    idxs = sorted(int(m.group(1)) for f in files
                  if (m := pattern.match(f)))
    if len(idxs) < 2:
        print("Need at least two checkpoints to compare.")
        return

    latest = idxs[-1]
    print(f"Latest checkpoint: {latest}")
    policy_new, _ = load_model(args.dir, latest)

    for prev in idxs[:-1]:
        print(f"\nEvaluating {latest} vs {prev} over {args.games} games…")
        policy_old, _ = load_model(args.dir, prev)
        wins, losses, draws = evaluate_pair(policy_new, policy_old, args.games, args.workers)

        total = wins + losses + draws
        print(f"  Wins:   {wins}/{total} ({wins/total:.2%})")
        print(f"  Losses: {losses}/{total} ({losses/total:.2%})")
        print(f"  Draws:  {draws}/{total} ({draws/total:.2%})")

if __name__ == "__main__":
    main()
