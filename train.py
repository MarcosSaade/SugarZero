# train.py

import argparse
import os
import sys
import random
import json
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from tqdm import tqdm
import matplotlib.pyplot as plt

import constants
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
NUM_SELF_PLAY_GAMES      = 5000
UCT_WARMUP_GAMES         = 400     # increased per discussion
UCT_SIMULATIONS          = 200
MCTS_SIMULATIONS         = 600
MIN_MCTS_SIMULATIONS     = 200
BATCH_SIZE               = 64
REPLAY_BUFFER_CAPACITY   = 10000
LEARNING_RATE            = 1e-3
WEIGHT_DECAY             = 1e-4
CHECKPOINT_INTERVAL      = 500
EPISODES_PER_TRAIN_BATCH = 10
DEVICE                   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MAX_MOVES                = 200
EVAL_RANDOM_GAMES        = 50
WARMUP_TRAIN_STEPS       = 1000

# filenames for persistent logs
LOSS_LOG_FILENAME    = "loss_history.json"
EVAL_LOG_FILENAME    = "eval_history.json"
ENTROPY_LOG_FILENAME = "entropy_history.json"


def save_json(data, path):
    with open(path, 'w') as f:
        json.dump(data, f)


def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)


def scheduled_simulations(game_idx: int) -> int:
    frac = (game_idx - 1) / (NUM_SELF_PLAY_GAMES - 1)
    sims = MIN_MCTS_SIMULATIONS + frac * (MCTS_SIMULATIONS - MIN_MCTS_SIMULATIONS)
    return int(sims)


def generate_self_play_data(policy_model, value_model, sim_count: int):
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
    game = GameState()
    history = []

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
    Eval vs random: disable exploration noise & temp during eval.
    """
    # temporarily mute randomness
    orig_eps      = constants.NOISE_EPSILON
    orig_temp_th  = constants.TEMP_MOVES_THRESHOLD
    constants.NOISE_EPSILON    = 0.0
    constants.TEMP_MOVES_THRESHOLD = 0

    def _eval_dummy_value(states, turns):
        bsz = states.size(0)
        return torch.full((bsz, 1), 0.5, device=DEVICE)

    def one_game(starting_player: int):
        game = GameState()
        while not game.game_over and game.move_count < MAX_MOVES:
            if game.turn == starting_player:
                move, _ = puct_search_with_policy(
                    game,
                    sim_count=MIN_MCTS_SIMULATIONS,
                    policy_model=policy_model,
                    value_model=_eval_dummy_value,
                    cpuct=EXPLORATION_WEIGHT,
                    device=DEVICE
                )
            else:
                moves = game.get_valid_moves()
                move = random.choice(moves) if moves else None
            game.make_move(*move)
        return game.winner == starting_player

    wins = 0
    with ThreadPoolExecutor(max_workers=workers) as exec:
        futures = [exec.submit(one_game, i % 2) for i in range(num_games)]
        for f in as_completed(futures):
            if f.result():
                wins += 1

    # restore
    constants.NOISE_EPSILON    = orig_eps
    constants.TEMP_MOVES_THRESHOLD = orig_temp_th

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


def compute_policy_entropy(policy_model, replay_buffer, batch_size=64):
    if len(replay_buffer) < batch_size:
        return None
    states, turns, _, _ = replay_buffer.sample(batch_size)
    states, turns = states.to(DEVICE), turns.to(DEVICE)
    with torch.no_grad():
        probs = policy_model(states, turns)
        logp  = torch.log(probs + 1e-8)
        entropy = -(probs * logp).sum(dim=1).mean().item()
    return entropy


def save_checkpoint(policy_model, value_model, game_idx, output_dir="."):
    os.makedirs(output_dir, exist_ok=True)
    torch.save(policy_model.state_dict(),
               os.path.join(output_dir, f'policy_model_{game_idx}.pt'))
    torch.save(value_model.state_dict(),
               os.path.join(output_dir, f'value_model_{game_idx}.pt'))
    print(f"Saved checkpoint at game {game_idx}")


def plot_curve(history, output_dir, filename, xlabel, ylabel, title, label):
    plt.figure()
    plt.plot(history, label=label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Self-play training for SugarZero")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--policy-path", type=str, default="")
    parser.add_argument("--value-path",  type=str, default="")
    parser.add_argument("--start-game", type=int, default=1)
    parser.add_argument("--output-dir", type=str,  default="./checkpoints")
    parser.add_argument("--workers",    type=int,  default=os.cpu_count())
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    # load or init logs
    loss_path    = os.path.join(args.output_dir, LOSS_LOG_FILENAME)
    eval_path    = os.path.join(args.output_dir, EVAL_LOG_FILENAME)
    entropy_path = os.path.join(args.output_dir, ENTROPY_LOG_FILENAME)

    loss_history    = load_json(loss_path)    if args.resume and os.path.exists(loss_path)    else []
    eval_history    = load_json(eval_path)    if args.resume and os.path.exists(eval_path)    else []
    entropy_history = load_json(entropy_path) if args.resume and os.path.exists(entropy_path) else []

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

    warmup_trained = False
    current_game   = args.start_game

    try:
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as executor:
            while current_game <= NUM_SELF_PLAY_GAMES:
                # after warm-up, train extra on UCT data
                if not warmup_trained and current_game > UCT_WARMUP_GAMES:
                    print(f"\n=== Training on warm-up data for {WARMUP_TRAIN_STEPS} steps ===")
                    for step in range(1, WARMUP_TRAIN_STEPS + 1):
                        loss = train_step(policy_model, value_model, optimizer, replay_buffer)
                        if loss is not None:
                            loss_history.append(loss)
                            if step % 100 == 0:
                                print(f"[Warmup Train] Step {step}, Loss: {loss:.4f}")
                    warmup_trained = True
                    print("=== Warm-up training complete, resuming self-play ===\n")

                batch_end = min(current_game + EPISODES_PER_TRAIN_BATCH - 1,
                                NUM_SELF_PLAY_GAMES)

                # generate self-play or UCT data
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

                # collect
                for future in as_completed(futures):
                    data = future.result()
                    for state, turn, policy_t, value in data:
                        replay_buffer.push(state, turn, policy_t, value)
                    pbar.update(1)

                # train on new data
                for idx in range(current_game, batch_end + 1):
                    loss = train_step(policy_model, value_model, optimizer, replay_buffer)
                    if loss is not None:
                        loss_history.append(loss)
                        print(f"[Train] Game {idx}, Loss: {loss:.4f}")
                        # policy entropy
                        ent = compute_policy_entropy(policy_model, replay_buffer)
                        if ent is not None:
                            entropy_history.append(ent)
                            print(f"[Entropy] After Game {idx}, Entropy: {ent:.4f}")

                # checkpoint / eval / plot
                if batch_end % CHECKPOINT_INTERVAL == 0 or batch_end == NUM_SELF_PLAY_GAMES:
                    save_checkpoint(policy_model, value_model, batch_end, args.output_dir)
                    # persist logs
                    save_json(loss_history, loss_path)
                    save_json(eval_history, eval_path)
                    save_json(entropy_history, entropy_path)
                    # plots
                    plot_curve(loss_history, args.output_dir,
                               "loss_curve.png", "Training Steps", "Loss",
                               "Training Loss Over Time", "Loss")
                    plot_curve(eval_history, args.output_dir,
                               "winrate_curve.png", "Eval (x500 games)", "Win Rate",
                               "Win Rate Over Time vs Random", "WinRate")
                    plot_curve(entropy_history, args.output_dir,
                               "entropy_curve.png", "Training Steps", "Policy Entropy",
                               "Policy Entropy Over Time", "Entropy")

                current_game = batch_end + 1

        pbar.close()

    except KeyboardInterrupt:
        print("\nInterrupted! Saving checkpoint & logs...")
        last_game = max(current_game - 1, args.start_game)
        save_checkpoint(policy_model, value_model, last_game, args.output_dir)
        # persist logs
        save_json(loss_history, loss_path)
        save_json(eval_history, eval_path)
        save_json(entropy_history, entropy_path)
        # final plots
        plot_curve(loss_history, args.output_dir,
                   "loss_curve.png", "Training Steps", "Loss",
                   "Training Loss Over Time", "Loss")
        plot_curve(eval_history, args.output-dir,
                   "winrate_curve.png", "Eval (x500 games)", "Win Rate",
                   "Win Rate Over Time vs Random", "WinRate")
        plot_curve(entropy_history, args.output-dir,
                   "entropy_curve.png", "Training Steps", "Policy Entropy",
                   "Policy Entropy Over Time", "Entropy")
        pbar.close()
        sys.exit(1)


if __name__ == "__main__":
    main()
