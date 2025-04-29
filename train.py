# train.py

import argparse
import os
import sys
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
from mcts import puct_search_with_policy
from policy_net import PolicyNet
from value_net import ValueNet
from replay_buffer import ReplayBuffer
from game_state import GameState
from utils import encode_board, sample_with_temperature

# ---------- Tame PyTorch threading ----------
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

# Hyperparameters
NUM_SELF_PLAY_GAMES      = 1000
MCTS_SIMULATIONS         = 400   # maximum simulations per move
MIN_MCTS_SIMULATIONS     = 100   # starting simulations per move
BATCH_SIZE               = 64
REPLAY_BUFFER_CAPACITY   = 10000
LEARNING_RATE            = 1e-3
CHECKPOINT_INTERVAL      = 100
EPISODES_PER_TRAIN_BATCH = 10
DEVICE                   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MAX_MOVES                = 200

def scheduled_simulations(game_idx: int) -> int:
    frac = (game_idx - 1) / (NUM_SELF_PLAY_GAMES - 1)
    sims = MIN_MCTS_SIMULATIONS + frac * (MCTS_SIMULATIONS - MIN_MCTS_SIMULATIONS)
    return int(sims)

def generate_self_play_data(policy_model, value_model, sim_count: int):
    """
    Run one self-play episode and return training data:
    Returns list of (state_tensor, turn, policy_dist, value).
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

        # For early moves, sample with temperature
        if len(history) < TEMP_MOVES_THRESHOLD:
            move = sample_with_temperature(policy_dist, TEMPERATURE)

        board_tensor, turn = encode_board(game.board, game.turn)
        history.append((board_tensor, float(turn), policy_dist))
        game.make_move(*move)

    # Assign game outcome values
    if game.game_over:
        winner = game.winner
    else:
        winner = None  # draw

    output = []
    for state_tensor, turn, policy_dict in history:
        if winner is None:
            value = 0.5
        else:
            value = 1.0 if winner == turn else 0.0
        # convert policy dict to tensor aligned with 81 moves
        policy_tensor = torch.zeros(81, dtype=torch.float32)
        for (s, e), p in policy_dict.items():
            policy_tensor[s*9 + e] = p
        output.append((torch.from_numpy(state_tensor), turn, policy_tensor, value))
    return output

def train_step(policy_model, value_model, optimizer, replay_buffer):
    if getattr(replay_buffer, 'push_count', len(replay_buffer)) < BATCH_SIZE:
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
    torch.save(policy_model.state_dict(), os.path.join(output_dir, f'policy_model_{game_idx}.pt'))
    torch.save(value_model.state_dict(),  os.path.join(output_dir, f'value_model_{game_idx}.pt'))
    print(f"Saved checkpoints at game {game_idx}")

def plot_metrics(loss_history, winrate_history, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    # Loss curve
    plt.figure()
    plt.plot(loss_history, label='Training Loss')
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Time')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'loss_curve.png'))
    plt.close()

    # Win-rate curve
    plt.figure()
    plt.plot(winrate_history, label='Win Rate vs Random')
    plt.xlabel('Evaluation Episodes')
    plt.ylabel('Win Rate')
    plt.title('Win Rate Over Time')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'winrate_curve.png'))
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Parallel self-play training for SugarZero")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoints")
    parser.add_argument("--policy-path", type=str, default="", help="Policy checkpoint")
    parser.add_argument("--value-path",  type=str, default="", help="Value checkpoint")
    parser.add_argument("--start-game", type=int, default=1, help="Game index to start from")
    parser.add_argument("--output-dir", type=str,  default="./checkpoints", help="Where to save checkpoints and graphs")
    parser.add_argument("--workers",    type=int,  default=os.cpu_count(), help="Parallel workers")
    args = parser.parse_args()

    policy_model = PolicyNet().to(DEVICE)
    value_model  = ValueNet().to(DEVICE)
    optimizer    = optim.Adam(
        list(policy_model.parameters()) + list(value_model.parameters()),
        lr=LEARNING_RATE
    )

    # load if resuming
    if args.resume:
        if os.path.isfile(args.policy_path):
            policy_model.load_state_dict(torch.load(args.policy_path, map_location=DEVICE))
        if os.path.isfile(args.value_path):
            value_model.load_state_dict(torch.load(args.value_path, map_location=DEVICE))

    replay_buffer = ReplayBuffer(REPLAY_BUFFER_CAPACITY)
    ctx = mp.get_context("spawn")
    pbar = tqdm(total=NUM_SELF_PLAY_GAMES, desc="Self-play games")

    # histories for graphing
    loss_history = []
    winrate_history = []

    current_game = args.start_game
    try:
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as executor:
            while current_game <= NUM_SELF_PLAY_GAMES:
                batch_end = min(current_game + EPISODES_PER_TRAIN_BATCH - 1,
                                NUM_SELF_PLAY_GAMES)

                # Parallel self-play
                futures = {}
                for idx in range(current_game, batch_end + 1):
                    sims = scheduled_simulations(idx)
                    futures[ executor.submit(generate_self_play_data,
                                             policy_model, value_model, sims) ] = idx

                for future in as_completed(futures):
                    data = future.result()
                    for state, turn, policy_t, value in data:
                        replay_buffer.push(state, turn, policy_t, value)
                    pbar.update(1)

                # Training & record loss
                for idx in range(current_game, batch_end + 1):
                    loss = train_step(policy_model, value_model, optimizer, replay_buffer)
                    if loss is not None:
                        loss_history.append(loss)
                        print(f"Game {idx}/{NUM_SELF_PLAY_GAMES}, Loss: {loss:.4f}")

                # Evaluate against random agent every checkpoint interval
                if batch_end % CHECKPOINT_INTERVAL == 0 or batch_end == NUM_SELF_PLAY_GAMES:
                    # simple random-eval: winrate over 20 games
                    wins = 0
                    for _ in range(20):
                        # TODO: implement evaluate_vs_random(policy_model, value_model)
                        pass
                    winrate = wins / 20
                    winrate_history.append(winrate)
                    print(f"Win rate vs random at game {batch_end}: {winrate:.2f}")

                    save_checkpoint(policy_model, value_model, batch_end, args.output_dir)
                    plot_metrics(loss_history, winrate_history, args.output_dir)

                current_game = batch_end + 1

        pbar.close()

    except KeyboardInterrupt:
        print("\nInterrupted! Saving checkpoint and plotting metrics...")
        last_game = max(current_game - 1, args.start_game)
        save_checkpoint(policy_model, value_model, last_game, args.output_dir)
        plot_metrics(loss_history, winrate_history, args.output_dir)
        pbar.close()
        sys.exit(1)

if __name__ == "__main__":
    main()
