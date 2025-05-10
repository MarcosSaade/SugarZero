# value_net.py

import torch
import torch.nn as nn

class ValueNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(12, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(64*3*3 + 1, 128),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, board_tensor: torch.Tensor, turn: torch.Tensor) -> torch.Tensor:
        """
        board_tensor: (batch, 2,3,3,6)
        turn:         (batch,) values 0 or 1
        returns:      (batch,1) win probability
        """
        bsz = board_tensor.size(0)
        x = board_tensor.permute(0,1,4,2,3).reshape(bsz, 12, 3, 3)
        x = self.conv(x)                # → (batch,64,3,3)
        x = x.view(bsz, -1)             # → (batch,576)
        t = turn.view(bsz,1).float()    # → (batch,1)
        x = torch.cat([x, t], dim=1)    # → (batch,576+1)
        return self.head(x)             # → (batch,1)
