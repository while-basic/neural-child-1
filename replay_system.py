import random
import torch
from torch import nn

class ReplayOptimizer:
    def __init__(self, memory_capacity=10000, device=None):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        self.memory = []
        self.capacity = memory_capacity
        self.importance_weights = nn.Parameter(torch.ones(memory_capacity, device=self.device))
        self.decay_factor = 0.99

    def add_experience(self, experience):
        if len(self.memory) < self.capacity:
            self.memory.append(experience)
        else:
            idx = random.randint(0, self.capacity - 1)
            self.memory[idx] = experience

    def sample_batch(self, batch_size=32):
        # Use the available number of experiences if fewer than the requested batch size.
        actual_batch_size = min(batch_size, len(self.memory))
        if actual_batch_size == 0:
            return [], []
        indices = random.sample(range(len(self.memory)), actual_batch_size)
        samples = [self.memory[i] for i in indices]
        return samples, indices

    def update_weights(self, indices, losses: torch.Tensor):
        self.importance_weights.data *= self.decay_factor
        for i, loss in zip(indices, losses):
            self.importance_weights.data[i] += loss
        if len(self.memory) > self.capacity * 0.95:
            prune_idx = torch.argsort(self.importance_weights)[:len(self.memory) // 20]
            self.memory = [m for i, m in enumerate(self.memory) if i not in prune_idx]
            self.importance_weights = nn.Parameter(
                torch.ones(len(self.memory), device=self.device),
                requires_grad=True
            )
