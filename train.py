# train.py

import argparse
import os
import sys
import random
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import matplotlib.pyplot as plt

from constants import (
    EXPLORATION_WEIGHT,
    TEMP_MOVES_THRESHOLD,
    TEMPERATURE
)
from mcts import puct_search, puct_search_with_policy
from policy_net import PolicyNet
from value_net import ValueNet
from replay_buffer import ReplayBuffer
from game_state import GameState
from utils import encode_board, sample_with_temperature

# ---------- Tame PyTorch threading ----------
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

# Hyperparameters
NUM_SELF_PLAY_GAMES      = 5000    # total self-play games
UCT_WARMUP_GAMES         = 200     # games vs UCT before NN guidance
UCT_SIMULATIONS          = 150     # sims for warmup UCT
MCTS_SIMULATIONS         = 400     # maximum simulations per move
MIN_MCTS_SIMULATIONS     = 100     # starting simulations per move
BATCH_SIZE               = 64
REPLAY_BUFFER_CAPACITY   = 10000
LEARNING_RATE            = 1e-3
WEIGHT_DECAY             = 1e-4    # L2 regularization
CHECKPOINT_INTERVAL      = 500     # save checkpoint & eval every this many games
EPISODES_PER_TRAIN_BATCH = 10
DEVICE                   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MAX_MOVES                = 200
EVAL_RANDOM_GAMES        = 50      # games to eval vs random

def scheduled_simulations(game_idx: int) -> int:
    frac = (game_idx - 1) / (NUM_SELF_PLAY_GAMES - 1)
    sims = MIN_MCTS_SIMULATIONS + frac * (MCTS_SIMULATIONS - MIN_MCTS_SIMULATIONS)
    return int(sims)

def generate_self_play_data(policy_model, value_model, sim_count: int):
    """
    Self-play using NN guidance and PUCT.
    """
    game = GameState()
    history = []

    while not game.game_over and game.move_count < MAX_MOVES:
        move, policy_dist = puct_search_with_policy(
            game,
            sim_count=sim_count,
            policy_model=policy_model,
            value_model=value_model,
            cpuct=EXPLORATION_WEIGHT,
            device=DEVICE
        )

        if len(history) < TEMP_MOVES_THRESHOLD:
            move = sample_with_temperature(policy_dist, TEMPERATURE)

        board_tensor, turn = encode_board(game.board, game.turn)
        history.append((board_tensor, float(turn), policy_dist))
        game.make_move(*move)

    winner = game.winner if game.game_over else None

    data = []
    for state_tensor, turn, policy_dict in history:
        value = 0.5 if winner is None else (1.0 if winner == turn else 0.0)
        policy_tensor = torch.zeros(81, dtype=torch.float32)
        for (s, e), p in policy_dict.items():
            policy_tensor[s * 9 + e] = p
        data.append((torch.from_numpy(state_tensor), turn, policy_tensor, value))
    return data

def generate_uct_data(sim_count: int):
    """
    Warmup self-play using pure UCT (no NN).
    Record one-hot policy and outcome values.
    """
    game = GameState()
    history = []

    # dummy value model always returns 0.5
    def _dummy_value(states, turns):
        bsz = states.size(0)
        return torch.full((bsz, 1), 0.5, device=DEVICE)

    while not game.game_over and game.move_count < MAX_MOVES:
        move = puct_search(
            game,
            sim_count=sim_count,
            policy_model=None,
            value_model=_dummy_value,
            cpuct=EXPLORATION_WEIGHT,
            device=DEVICE
        )

        board_tensor, turn = encode_board(game.board, game.turn)
        history.append((board_tensor, float(turn), move))
        game.make_move(*move)

    winner = game.winner if game.game_over else None

    data = []
    for state_tensor, turn, move in history:
        value = 0.5 if winner is None else (1.0 if winner == turn else 0.0)
        policy_tensor = torch.zeros(81, dtype=torch.float32)
        policy_tensor[move[0] * 9 + move[1]] = 1.0
        data.append((torch.from_numpy(state_tensor), turn, policy_tensor, value))
    return data

def evaluate_vs_random(policy_model, num_games: int, workers: int):
    """
    Evaluate current policy_model vs uniform random mover.
    Returns win rate for policy_model playing first/second.
    """
    def one_game(starting_player: int):
        game = GameState()
        while not game.game_over and game.move_count < MAX_MOVES:
            if game.turn == starting_player:
                # policy move
                move, _ = puct_search_with_policy(
                    game,
                    sim_count=MIN_MCTS_SIMULATIONS,
                    policy_model=policy_model,
                    value_model=None,
                    cpuct=EXPLORATION_WEIGHT,
                    device=DEVICE
                )
            else:
                # random move
                moves = game.get_valid_moves()
                move = random.choice(moves) if moves else None
            game.make_move(*move)
        return game.winner == starting_player

    wins = 0
    with ProcessPoolExecutor(max_workers=workers) as exec:
        futures = [exec.submit(one_game, i % 2) for i in range(num_games)]
        for f in as_completed(futures):
            if f.result():
                wins += 1
    return wins / num_games

def train_step(policy_model, value_model, optimizer, replay_buffer):
    if replay_buffer.push_count < BATCH_SIZE:
        return None

    policy_model.train()
    value_model.train()

    states, turns, target_policies, target_values = replay_buffer.sample(BATCH_SIZE)
    states, turns = states.to(DEVICE), turns.to(DEVICE)
    target_policies, target_values = target_policies.to(DEVICE), target_values.to(DEVICE)

    pred_policies = policy_model(states, turns)
    pred_values   = value_model(states, turns)

    policy_loss = F.kl_div(pred_policies.log(), target_policies, reduction='batchmean')
    value_loss  = F.mse_loss(pred_values, target_values)
    loss = policy_loss + value_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()

def save_checkpoint(policy_model, value_model, game_idx, output_dir="."):
    os.makedirs(output_dir, exist_ok=True)
    torch.save(policy_model.state_dict(),
               os.path.join(output_dir, f'policy_model_{game_idx}.pt'))
    torch.save(value_model.state_dict(),
               os.path.join(output_dir, f'value_model_{game_idx}.pt'))
    print(f"Saved checkpoint at game {game_idx}")

def plot_loss_curve(loss_history, output_dir):
    plt.figure()
    plt.plot(loss_history, label='Training Loss')
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Time')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'loss_curve.png'))
    plt.close()

def plot_eval_curve(eval_history, output_dir):
    plt.figure()
    plt.plot(eval_history, label='Win Rate vs Random')
    plt.xlabel(f'Evaluation (every {CHECKPOINT_INTERVAL} games)')
    plt.ylabel('Win Rate')
    plt.title('Win Rate Over Time')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'winrate_curve.png'))
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Self-play training for SugarZero")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoints")
    parser.add_argument("--policy-path", type=str, default="", help="Policy checkpoint path")
    parser.add_argument("--value-path",  type=str, default="", help="Value checkpoint path")
    parser.add_argument("--start-game", type=int, default=1, help="Game index to start from")
    parser.add_argument("--output-dir", type=str,  default="./checkpoints", help="Directory to save outputs")
    parser.add_argument("--workers",    type=int,  default=os.cpu_count(), help="Parallel workers")
    args = parser.parse_args()

    policy_model = PolicyNet().to(DEVICE)
    value_model  = ValueNet().to(DEVICE)
    optimizer    = optim.Adam(
        list(policy_model.parameters()) + list(value_model.parameters()),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    if args.resume:
        if os.path.isfile(args.policy_path):
            policy_model.load_state_dict(torch.load(args.policy_path, map_location=DEVICE))
        if os.path.isfile(args.value_path):
            value_model.load_state_dict(torch.load(args.value_path, map_location=DEVICE))

    replay_buffer = ReplayBuffer(REPLAY_BUFFER_CAPACITY)
    ctx = mp.get_context("spawn")
    pbar = tqdm(total=NUM_SELF_PLAY_GAMES, desc="Self-play games")

    loss_history = []
    eval_history = []
    current_game = args.start_game

    try:
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as executor:
            while current_game <= NUM_SELF_PLAY_GAMES:
                batch_end = min(current_game + EPISODES_PER_TRAIN_BATCH - 1,
                                NUM_SELF_PLAY_GAMES)

                # choose warmup or guided self-play
                futures = {}
                for idx in range(current_game, batch_end + 1):
                    if idx <= UCT_WARMUP_GAMES:
                        futures[executor.submit(generate_uct_data, UCT_SIMULATIONS)] = idx
                    else:
                        sims = scheduled_simulations(idx)
                        futures[executor.submit(
                            generate_self_play_data,
                            policy_model, value_model, sims
                        )] = idx

                # collect self-play data
                for future in as_completed(futures):
                    data = future.result()
                    for state, turn, policy_t, value in data:
                        replay_buffer.push(state, turn, policy_t, value)
                    pbar.update(1)

                # training steps
                for idx in range(current_game, batch_end + 1):
                    loss = train_step(policy_model, value_model, optimizer, replay_buffer)
                    if loss is not None:
                        loss_history.append(loss)
                        print(f"[Train] Game {idx}, Loss: {loss:.4f}")

                # periodic checkpoint, eval, plotting
                if batch_end % CHECKPOINT_INTERVAL == 0 or batch_end == NUM_SELF_PLAY_GAMES:
                    save_checkpoint(policy_model, value_model, batch_end, args.output_dir)
                    plot_loss_curve(loss_history, args.output_dir)

                    winrate = evaluate_vs_random(policy_model, EVAL_RANDOM_GAMES, args.workers)
                    eval_history.append(winrate)
                    print(f"[Eval] Win rate vs random at game {batch_end}: {winrate:.2f}")
                    plot_eval_curve(eval_history, args.output_dir)

                current_game = batch_end + 1

        pbar.close()

    except KeyboardInterrupt:
        print("\nInterrupted! Saving last checkpoint and plots...")
        last_game = max(current_game - 1, args.start_game)
        save_checkpoint(policy_model, value_model, last_game, args.output_dir)
        plot_loss_curve(loss_history, args.output_dir)
        plot_eval_curve(eval_history, args.output_dir)
        pbar.close()
        sys.exit(1)

if __name__ == "__main__":
    main()
