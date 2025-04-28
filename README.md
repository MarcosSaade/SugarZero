# SugarZero

**SugarZero** is a reinforcement learning playground for the table game *Sugar*, which I'm actively developing.  
It was motivated both by a desire to understand how AlphaZero works, and by my wish to create a strong opponent to play against.

---

### How *Sugar* Works
Players control stacks of pieces across a 3×3 board.  
Each turn, a player moves a piece from one square to an adjacent square. Movement is constrained by stack heights — you can only move onto a shorter or equal stack.  
Additionally, during the opening moves, players must include at least one forward movement.  
The goal is to leave your opponent without any valid moves.

---

### About the Name
*Sugar* is named after how I originally invented and played the game — using sugar packets at restaurant tables with friends.

---

### Features
- Custom Monte Carlo Tree Search (MCTS) implementation with PUCT (policy and value guidance).
- Simple policy and value neural networks.
- Playable graphical interface built with Pygame.
- Fast MCTS simulations using threading for parallelism.
- Human vs. AI gameplay with adjustable difficulty.
- Self-play data generation for training.
- Board encoding for neural network input.
- Value network estimating win probabilities from board positions.
- Policy network predicting promising moves.
- Periodic checkpoint saving during training.

---

### Recent Additions
- Parallelized self-play for faster training.
- Batch evaluation of board states to improve neural network efficiency.
- Replay buffer for sampling training batches.
- Support for loading and resuming from model checkpoints.

---

### Main Files
- `game.py` — Pygame interface and game logic.
- `game_state.py` — Lightweight board representation and rules.
- `mcts.py` — MCTS agent with PUCT exploration and network guidance.
- `policy_net.py` — Feedforward neural network predicting moves.
- `value_net.py` — Feedforward neural network estimating outcomes.
- `train.py` — Self-play training loop.
- `replay_buffer.py` — Storage and sampling of training examples.
- `utils.py`, `constants.py` — Utility functions and configuration settings.

---

### How to Run
```bash
python main.py
```
