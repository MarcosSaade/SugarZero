# policy_net.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class PolicyNet(nn.Module):
    def __init__(self):
        super(PolicyNet, self).__init__()
        # 108 board features + 1 turn feature → 81 possible moves
        self.net = nn.Sequential(
            nn.Linear(108 + 1, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 81)           # one logit per (start,end) pair
        )

    def forward(self, board_tensor: torch.Tensor, turn: torch.Tensor) -> torch.Tensor:
        """
        board_tensor: shape (batch, 2,3,3,6), dtype=float32
        turn:        shape (batch,), values 0 or 1, dtype=float32 or long
        returns:     shape (batch, 81), probability distribution over moves
        """
        bsz = board_tensor.size(0)
        x = board_tensor.view(bsz, -1)            # → (batch,108)
        t = turn.view(bsz, 1).float()             # → (batch,1)
        x = torch.cat([x, t], dim=1)              # → (batch,109)
        logits = self.net(x)                      # → (batch,81)
        probs = F.softmax(logits, dim=1)          # normalize to sum=1
        return probs
