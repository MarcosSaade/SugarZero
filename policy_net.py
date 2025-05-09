# policy_net.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class PolicyNet(nn.Module):
    def __init__(self):
        super().__init__()
        # input channels = 2 players × 6 height‐levels = 12
        self.conv = nn.Sequential(
            nn.Conv2d(12, 64, kernel_size=3, padding=1),  # → (64,3,3)
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),  # → (64,3,3)
            nn.ReLU(),
        )
        # flatten + turn feature → hidden → 81 logits
        self.head = nn.Sequential(
            nn.Linear(64*3*3 + 1, 128),
            nn.ReLU(),
            nn.Linear(128, 81)
        )

    def forward(self, board_tensor: torch.Tensor, turn: torch.Tensor) -> torch.Tensor:
        """
        board_tensor: (batch, 2,3,3,6)
        turn:         (batch,) values 0 or 1
        returns:      (batch,81) probability distribution over moves
        """
        bsz = board_tensor.size(0)
        # reshape to (batch,12,3,3)
        x = board_tensor.permute(0,1,4,2,3).reshape(bsz, 12, 3, 3)
        x = self.conv(x)                # → (batch,64,3,3)
        x = x.view(bsz, -1)             # → (batch, 64*3*3)
        t = turn.view(bsz,1).float()    # → (batch,1)
        x = torch.cat([x, t], dim=1)    # → (batch, 64*3*3+1)
        logits = self.head(x)           # → (batch,81)
        return F.softmax(logits, dim=1)
