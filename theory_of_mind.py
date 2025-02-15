import torch
import torch.nn as nn

class TheoryOfMind(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Perspective taking capability
        self.perspective_network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim)
        )
        
        # Belief modeling
        self.belief_network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim)
        )
        
        # Social bias learning
        self.social_bias = nn.Parameter(torch.zeros(hidden_dim))
        
        # Mental state attribution
        self.attribution_network = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def forward(self, x):
        # Take others' perspective
        perspective = self.perspective_network(x)
        
        # Model others' beliefs
        beliefs = self.belief_network(x)
        
        # Apply social learning bias
        social_context = x + self.social_bias
        
        # Attribute mental states
        combined = torch.cat([perspective, beliefs], dim=-1)
        mental_state = self.attribution_network(combined)
        
        # Combine all aspects
        output = mental_state + social_context
        
        return output
    
    def get_state(self):
        return {
            'social_bias': self.social_bias.tolist(),
            'perspective_taking': self.perspective_network[0].weight.mean().item(),
            'belief_modeling': self.belief_network[0].weight.mean().item()
        }
