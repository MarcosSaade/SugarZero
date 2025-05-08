# replay_buffer.py

import random
from collections import deque
import torch

class ReplayBuffer:
    def __init__(self, capacity, mix_ratios=None):
        """
        mix_ratios: dict with keys "uct", "random", "self" summing to 1.0
        e.g. {"uct":0.2, "random":0.1, "self":0.7}
        """
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.push_count = 0
        # default mixing: 20% UCT, 10% random, 70% self-play
        self.mix = mix_ratios or {"uct":0.2, "random":0.1, "self":0.7}

    def push(self, state, turn, policy, value, origin="self"):
        """
        origin: one of "uct", "random", "self"
        """
        self.push_count += 1
        # store a 5-tuple
        self.buffer.append((state, turn, policy, value, origin))

    def sample(self, batch_size):
        n_uct    = int(batch_size * self.mix["uct"])
        n_rand   = int(batch_size * self.mix["random"])
        n_self   = batch_size - n_uct - n_rand

        # partition buffer by origin
        uct_items    = [x for x in self.buffer if x[4]=="uct"]
        rand_items   = [x for x in self.buffer if x[4]=="random"]
        self_items   = [x for x in self.buffer if x[4]=="self"]

        batch = []
        # helper to sample with fallback
        def take(src, n):
            if len(src) >= n:
                return random.sample(src, n)
            else:
                # not enough of that origin, fill from entire buffer
                taken = list(src)
                taken += random.choices(self.buffer, k=n-len(src))
                return taken

        batch += take(uct_items,  n_uct)
        batch += take(rand_items, n_rand)
        batch += take(self_items, n_self)

        # if for some reason batch smaller
        if len(batch) < batch_size:
            batch += random.choices(self.buffer, k=batch_size - len(batch))

        states, turns, policies, values, _ = zip(*batch)
        states   = torch.stack(states)
        turns    = torch.tensor(turns, dtype=torch.float32).unsqueeze(1)
        policies = torch.stack(policies)
        values   = torch.tensor(values, dtype=torch.float32).unsqueeze(1)
        return states, turns, policies, values

    def __len__(self):
        return len(self.buffer)
