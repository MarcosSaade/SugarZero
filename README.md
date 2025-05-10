# SugarZero

**SugarZero** is an AlphaZero-inspired reinforcement learning AI for the abstract strategy game *Sugar*. It combines neural networks with Monte Carlo Tree Search to learn optimal play through self-competition, without human knowledge or heuristics.

This project was motivated by my fascination with AlphaZero's approach and my desire to truly understand how it works by building a simplified version that could run on consumer hardware.

![Screenshot from 2025-05-10 07-31-14](https://github.com/user-attachments/assets/4bdc27d0-71ad-4044-8e00-e7db6911a74d)
---

## The Game: Sugar

*Sugar* is a compact abstract strategy game played on a 3×3 board:

- Players control 6 stackable pieces, initially placed on their back rank
- On each turn, a player moves one piece from the top of a stack to an adjacent square (horizontally, vertically, or diagonally)
- A piece can only move to a square if the destination stack is shorter than or equal in height to the stack it's moving from
- For the first three moves, each player must make at least one "forward" move (toward the opponent's side)
- The game ends when a player cannot make any legal moves, and that player loses

> **About the Name**: I originally invented and played the game using sugar packets at restaurant tables with friends, hence the name "Sugar".

---

## Technical Features

### Core Architecture
- **Neural Networks**: Combined policy (move selection) and value (position evaluation) networks
- **Monte Carlo Tree Search**: Enhanced with PUCT
- **Self-Play Training**: AI improves by generating its own training data through self-competition
- **Parallel Processing**: Multithreaded game simulation using Python's `concurrent.futures`

### Training Enhancements
- **Dirichlet Noise**: Added to MCTS root node to encourage exploration (α=0.8)
- **Temperature Sampling**: Controls move diversity through training (T=1.0 → 0.5)
- **Draw Filtering**: Prevents learning from cycles by discarding excessively long games
- **Winner Oversampling**: Prioritizes learning from decisive victories
- **Replay Buffer**: Stores and samples 15,000 game positions for efficient training

### User Interface
- **Interactive Gameplay**: Human vs. AI matches with adjustable difficulty
---

## Key Components

- `game_state.py` — Efficient board representation and rules implementation
- `mcts.py` — Monte Carlo Tree Search with PUCT exploration
- `neural_net.py` — CNN for combined policy and value predictions
- `replay_buffer.py` — Storage and sampling of training examples
- `train.py` — Self-play generation and network training
- `game.py` — Pygame UI for human play against the AI

---

## Getting Started

### Requirements
```
python >= 3.8
pytorch >= 1.7.0
numpy
pygame
```

### Installation
```bash
git clone https://github.com/MarcosSaade/SugarZero.git
cd SugarZero
pip install -r requirements.txt
```

### Running the Game
```bash
python main.py
```

### Controls
- **R**: Restart game
- **Q**: Quit
- **A**: Toggle AI on/off
- **S**: Swap which side the AI plays
- **D**: Cycle AI difficulty (Easy → Medium → Hard)

