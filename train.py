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

from constants import EXPLORATION_WEIGHT
from mcts import puct_search_with_policy
from policy_net import PolicyNet
from value_net import ValueNet
from replay_buffer import ReplayBuffer
from game_state import GameState
from utils import encode_board

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
CHECKPOINT_INTERVAL      = 100   # games between automatic saves
EPISODES_PER_TRAIN_BATCH = 10    # self-play games per training batch
DEVICE                   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MAX_MOVES                = 200   # cap moves per self-play to detect draws


def scheduled_simulations(game_idx: int) -> int:
    """
    Linearly interpolate number of MCTS simulations from MIN to MAX
    across the training schedule.
    """
    frac = (game_idx - 1) / (NUM_SELF_PLAY_GAMES - 1)
    sims = MIN_MCTS_SIMULATIONS + frac * (MCTS_SIMULATIONS - MIN_MCTS_SIMULATIONS)
    return int(sims)


def generate_self_play_data(policy_model, value_model, sim_count: int):
    """
    Run one self-play episode and return training data instead of pushing to buffer.
    Returns a list of tuples: (state_tensor_cpu, turn, policy_tensor_cpu, value)
    """
    game = GameState()
    history = []

    # play until game over or max moves
    while not game.game_over and game.move_count < MAX_MOVES:
        move, policy_dist = puct_search_with_policy(
            game,
            sim_count=sim_count,
            policy_model=policy_model,
            value_model=value_model,
            cpuct=EXPLORATION_WEIGHT,
            device=DEVICE
        )

        board_tensor, turn = encode_board(game.board, game.turn)
        state_tensor = torch.from_numpy(board_tensor)
        policy_tensor = torch.zeros(81, dtype=torch.float32)
        for (s, e), p in policy_dist.items():
            policy_tensor[s * 9 + e] = p

        history.append((state_tensor, float(turn), policy_tensor))
        game.make_move(*move)

    # Determine winner or draw
    if game.game_over:
        winner = game.winner
    else:
        winner = None  # treat as draw

    output = []
    for state_tensor, turn, policy_tensor in history:
        if winner is None:
            value = 0.5
        else:
            value = 1.0 if winner == turn else 0.0
        output.append((state_tensor, turn, policy_tensor, value))
    return output


def train_step(policy_model, value_model, optimizer, replay_buffer):
    if len(replay_buffer) < BATCH_SIZE:
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
    policy_path = os.path.join(output_dir, f'policy_model_{game_idx}.pt')
    value_path  = os.path.join(output_dir, f'value_model_{game_idx}.pt')
    torch.save(policy_model.state_dict(), policy_path)
    torch.save(value_model.state_dict(), value_path)
    print(f"Saved checkpoints at game {game_idx}:\n  {policy_path}\n  {value_path}")


def main():
    parser = argparse.ArgumentParser(description="Parallel self-play training for SugarZero")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from given checkpoints and start-game index")
    parser.add_argument("--policy-path", type=str, default="",
                        help="Path to policy model checkpoint (.pt)")
    parser.add_argument("--value-path", type=str, default="",
                        help="Path to value model checkpoint (.pt)")
    parser.add_argument("--start-game", type=int, dest="start_game", default=1,
                        help="Which game index to start/resume from")
    parser.add_argument("--output-dir", type=str, dest="output_dir", default="./checkpoints",
                        help="Directory to save model checkpoints and plots")
    parser.add_argument("--workers", type=int, default=os.cpu_count(),
                        help="Number of parallel self-play workers (default = all cores)")
    args = parser.parse_args()

    policy_model = PolicyNet().to(DEVICE)
    value_model  = ValueNet().to(DEVICE)
    optimizer    = optim.Adam(
        list(policy_model.parameters()) + list(value_model.parameters()),
        lr=LEARNING_RATE
    )

    if args.resume:
        if args.policy_path and os.path.isfile(args.policy_path):
            policy_model.load_state_dict(torch.load(args.policy_path, map_location=DEVICE))
            print(f"Loaded policy model from {args.policy_path}")
        if args.value_path and os.path.isfile(args.value_path):
            value_model.load_state_dict(torch.load(args.value_path, map_location=DEVICE))
            print(f"Loaded value model from {args.value_path}")

    replay_buffer = ReplayBuffer(REPLAY_BUFFER_CAPACITY)
    loss_history = []

    ctx = mp.get_context("spawn")
    pbar = tqdm(total=NUM_SELF_PLAY_GAMES, desc="Self-play games")
    current_game = args.start_game
    try:
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as executor:
            while current_game <= NUM_SELF_PLAY_GAMES:
                batch_end = min(current_game + EPISODES_PER_TRAIN_BATCH - 1, NUM_SELF_PLAY_GAMES)

                # --- Parallel self-play ---
                futures = {}
                for idx in range(current_game, batch_end + 1):
                    sims = scheduled_simulations(idx)
                    print(f"Scheduling self-play game {idx} with {sims} sims")
                    future = executor.submit(
                        generate_self_play_data,
                        policy_model, value_model, sims
                    )
                    futures[future] = idx

                # collect results as they finish
                for future in as_completed(futures):
                    idx = futures[future]
                    data = future.result()
                    for state, turn, policy_t, value in data:
                        replay_buffer.push(state, turn, policy_t, value)
                    print(f"Completed self-play game {idx}, buffer size: {len(replay_buffer)}")
                    pbar.update(1)

                # --- Training batch ---
                for idx in range(current_game, batch_end + 1):
                    loss = train_step(policy_model, value_model, optimizer, replay_buffer)
                    if loss is not None:
                        loss_history.append((idx, loss))
                        print(f"Game {idx}/{NUM_SELF_PLAY_GAMES}, Loss: {loss:.4f}")

                # checkpoint
                if batch_end % CHECKPOINT_INTERVAL == 0 or batch_end == NUM_SELF_PLAY_GAMES:
                    save_checkpoint(policy_model, value_model, batch_end, output_dir=args.output_dir)

                current_game = batch_end + 1
        pbar.close()

        # After training, plot loss history
        if loss_history:
            games, losses = zip(*loss_history)
            plt.figure()
            plt.plot(games, losses)
            plt.xlabel("Game")
            plt.ylabel("Loss")
            plt.title("Training Loss over Time")
            plot_path = os.path.join(args.output_dir, "loss_plot.png")
            plt.savefig(plot_path)
            print(f"Saved loss plot to {plot_path}")
    except KeyboardInterrupt:
        print("\nInterrupted! Saving checkpoint...")
        last_game = max(current_game - 1, args.start_game)
        save_checkpoint(policy_model, value_model, last_game, output_dir=args.output_dir)
        pbar.close()
        print("Checkpoint saved. Exiting.")
        sys.exit(1)


if __name__ == "__main__":
    main()
