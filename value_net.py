# value_net.py

import torch
import torch.nn as nn

class ValueNet(nn.Module):
    def __init__(self):
        super(ValueNet, self).__init__()
        # flattened board has 2×3×3×6 = 108 features, +1 for turn
        self.net = nn.Sequential(
            nn.Linear(108 + 1, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, board_tensor: torch.Tensor, turn: torch.Tensor) -> torch.Tensor:
        """
        board_tensor: shape (batch, 2,3,3,6), dtype=torch.float32
        turn:        shape (batch,), values 0 or 1, dtype=torch.float32 or torch.long
        returns:     shape (batch, 1), win probability for the player to move
        """
        bsz = board_tensor.size(0)
        x = board_tensor.view(bsz, -1)           # → (batch,108)
        t = turn.view(bsz, 1).float()            # → (batch,1)
        x = torch.cat([x, t], dim=1)             # → (batch,109)
        return self.net(x)                       # → (batch,1)
