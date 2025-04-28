# SugarZero

**SugarZero** is a reinforcement learning playground for the table game *Sugar*, which I'm actively developing.  
It currently features:
- A custom implementation of Monte Carlo Tree Search (MCTS) with UCB exploration.
- A simple policy/value neural network architecture (currently untrained).
- A playable graphical interface built with Pygame.

The project supports:
- Human vs. AI gameplay, with adjustable difficulty.
- Fast MCTS simulations with threading optimizations.
- Board encoding for neural network training.
- A value network estimating win probabilities from board positions.

**Main files:**
- `game.py`: Pygame interface and game logic.
- `mcts.py`: MCTS agent for AI decision-making.
- `value_net.py`: Simple feedforward neural network for value estimation.
- `game_state.py`: Lightweight board representation for efficient cloning.
- `utils.py`, `constants.py`: Utility functions and configuration.

**Run the game:**
```bash
python main.py
```

**Next steps:**
- Train the value network to improve AI strength.
- Integrate the policy network into move selection.

---
*This project is still in development!*
