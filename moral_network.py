import torch
from torch import nn
from typing import Dict

class MoralPolicyNetwork(nn.Module):
    def __init__(self, input_dim=128, device=torch.device('cuda')):
        super().__init__()
        self.device = device
        
        # Input projection layer: projects from input_dim (default 128) to 256
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.GELU()
        )
        
        # Ethical encoder: processes the projected features into a 256-dimensional representation
        self.ethical_encoder = nn.Sequential(
            nn.Linear(256, 512),  # Takes output from the projection layer
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, 256)
        )
        
        # Value head: produces the final moral value as a scalar
        self.value_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Tanh()
        )
        
        # Use a ParameterDict to ensure these parameters are registered and moved to the correct device.
        self.safety_filters = nn.ParameterDict({
            'self_preservation': nn.Parameter(torch.randn(256, device=self.device)),
            'social_norms': nn.Parameter(torch.randn(256, device=self.device))
        })
        
        # Move all module parameters to the specified device.
        self.to(self.device)

    def forward(self, thought_embedding: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Project the input into the proper dimension.
        projected = self.input_projection(thought_embedding)
        
        # Process the projected tensor through the ethical encoder.
        encoded = self.ethical_encoder(projected)
        
        # Apply each safety filter. Since safety_filters is now a ParameterDict,
        # its values will be on the same device (cuda) as encoded.
        for constraint in self.safety_filters.values():
            encoded = encoded * torch.sigmoid(constraint)
            
        # Compute the moral value from the filtered encoding.
        moral_value = self.value_head(encoded)
        
        return {
            'moral_score': moral_value,
            'constraint_applied': encoded
        }

    def reinforce(self, positive_examples, negative_examples):
        pos_scores = self(positive_examples)['moral_score']
        neg_scores = self(negative_examples)['moral_score']
        loss = torch.relu(1 - pos_scores + neg_scores).mean()
        return loss