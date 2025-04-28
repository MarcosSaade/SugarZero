# replay_buffer.py
import random
from collections import deque
import torch

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, turn, policy, value):
        """
        Args:
            state: torch.Tensor (2×3×3×6)
            turn: float (0.0 or 1.0)
            policy: torch.Tensor (81,)
            value: float (0.0 or 1.0)
        """
        self.buffer.append((state, turn, policy, value))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, turns, policies, values = zip(*batch)
        states = torch.stack(states)                   # (batch,2,3,3,6)
        turns = torch.tensor(turns, dtype=torch.float32).unsqueeze(1)
        policies = torch.stack(policies)               # (batch,81)
        values = torch.tensor(values, dtype=torch.float32).unsqueeze(1)
        return states, turns, policies, values

    def __len__(self):
        return len(self.buffer)
