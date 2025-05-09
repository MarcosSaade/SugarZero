# policy_net.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class PolicyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(12, 64, kernel_size=3, padding=1),   # → (64,3,3)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),   # → (64,3,3)
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(64*3*3 + 1, 128),
            nn.ReLU(),
            nn.Linear(128, 81)
        )

    def forward(self, board_tensor: torch.Tensor, turn: torch.Tensor) -> torch.Tensor:
        bsz = board_tensor.size(0)
        x = board_tensor.permute(0,1,4,2,3).reshape(bsz, 12, 3, 3)  # (B,12,3,3)
        x = self.conv(x)           # (B,64,3,3)
        x = x.view(bsz, -1)        # (B,576)
        t = turn.view(bsz,1).float()
        x = torch.cat([x, t], dim=1)
        logits = self.head(x)      # (B,81)
        return F.softmax(logits, dim=1)
