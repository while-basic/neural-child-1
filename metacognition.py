import torch
from torch import nn

class MetacognitionSystem(nn.Module):
    def __init__(self, base_dim=128, num_hypotheses=5):
        super().__init__()
        self.num_hypotheses = num_hypotheses
        self.base_network = nn.Sequential(
            nn.Linear(base_dim, 256),
            nn.GELU(),
            nn.Linear(256, base_dim)
        )
        self.hypothesis_network = nn.ModuleList([
            nn.Sequential(
                nn.Linear(base_dim, 256),
                nn.GELU(),
                nn.Linear(256, base_dim)
            ) for _ in range(num_hypotheses)
        ])
        self.critic = nn.Sequential(
            nn.Linear(base_dim * 2, 512),
            nn.GELU(),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )
        self.bayesian_layer = nn.LSTM(base_dim, base_dim)
        self.complexity_head = nn.Sequential(
            nn.Linear(base_dim, base_dim // 2),
            nn.GELU(),
            nn.Linear(base_dim // 2, 1),
            nn.Sigmoid()
        )
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, input_embedding):
        base_output = self.base_network(input_embedding)
        combined = torch.cat([input_embedding, base_output], dim=-1)
        confidence = self.critic(combined)
        _, (hidden, _) = self.bayesian_layer(base_output.unsqueeze(0))
        uncertainty = hidden.squeeze(0)
        complexity = self.complexity_head(base_output)
        return {
            'thought': base_output,
            'confidence': confidence,
            'uncertainty': uncertainty,
            'complexity': complexity
        }

    def self_correct(self, thought_embedding, temperature=0.7):
        alternatives = []
        scores = []
        base_uncertainty = self.forward(thought_embedding)['uncertainty']
        for i, net in enumerate(self.hypothesis_network):
            noise = torch.randn_like(thought_embedding) * temperature * (i + 1) / self.num_hypotheses
            alt = net(thought_embedding + noise)
            alternatives.append(alt)
            combined = torch.cat([thought_embedding, alt], -1)
            score = self.critic(combined)
            scores.append(score)
        if not alternatives:
            return thought_embedding
        weighted_scores = torch.stack(scores) * (1 - base_uncertainty)
        best_idx = torch.argmax(weighted_scores)
        return alternatives[best_idx]
