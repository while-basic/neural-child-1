import torch
import torch.nn as nn
from collections import deque

class TheoryOfMind(nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        self.device = device
        
        # Let's calculate the exact input dimension
        # base_dim (128) + sensory (256) + drives (10) + emotional (4) = 398
        input_dim = 398  # Updated to match the actual combined input size
        
        self.mental_state_predictor = nn.Sequential(
            nn.Linear(input_dim, 512),  # Update input dimension here
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 128)
        ).to(device)
        
        self.perspective_taking = nn.ModuleDict({
            'emotional': nn.Linear(128, 4),
            'belief': nn.Linear(128, 64),
            'intention': nn.Linear(128, 32),
            'attention': nn.Linear(128, 16)
        }).to(device)
        
        self.relationship_memory = deque(maxlen=1000)
        self.social_bias = nn.Parameter(torch.ones(4, device=device))
        
    def forward(self, social_context):
        # Add shape checking for debugging
        if social_context.dim() == 1:
            social_context = social_context.unsqueeze(0)
            
        mental_state = self.mental_state_predictor(social_context)
        predictions = {
            'emotional': torch.sigmoid(self.perspective_taking['emotional'](mental_state)),
            'belief': torch.tanh(self.perspective_taking['belief'](mental_state)),
            'intention': torch.softmax(self.perspective_taking['intention'](mental_state), dim=-1),
            'attention': torch.sigmoid(self.perspective_taking['attention'](mental_state))
        }
        predictions['emotional'] *= self.social_bias
        return predictions
    
    def update_relationship_model(self, interaction: torch.Tensor, outcome: float):
        self.relationship_memory.append((interaction, outcome))
        if len(self.relationship_memory) >= 100:
            recent_outcomes = torch.tensor([o for _, o in list(self.relationship_memory)[-100:]])
            self.social_bias.data = torch.sigmoid(recent_outcomes.mean() * self.social_bias)

class AttachmentSystem(nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        self.device = device
        # Attachment system expects a 4-dimensional emotional vector
        self.attachment_styles = nn.Parameter(
            torch.tensor([0.7, 0.1, 0.1, 0.1], device=device),
            requires_grad=True
        )
        self.trust_network = nn.Sequential(
            nn.Linear(4, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        ).to(device)
        self.bonding_network = nn.Sequential(
            nn.Linear(4, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 64)
        ).to(device)
        self.caregiving_history = deque(maxlen=1000)
        self.trust_level = nn.Parameter(torch.tensor(0.5, device=device))
        
    def forward(self, caregiver_input: torch.Tensor):
        trust_prediction = self.trust_network(caregiver_input)
        bonding_features = self.bonding_network(caregiver_input)
        self.trust_level.data = 0.95 * self.trust_level.data + 0.05 * trust_prediction
        attachment_response = torch.softmax(self.attachment_styles * self.trust_level, dim=0)
        return {
            'trust_level': self.trust_level,
            'attachment_style': attachment_response,
            'bonding_features': bonding_features
        }
    
    def update_attachment(self, interaction_quality: float):
        self.caregiving_history.append(interaction_quality)
        if len(self.caregiving_history) >= 100:
            recent_quality = torch.tensor(list(self.caregiving_history), device=self.device)
            quality_mean = recent_quality.mean()
            if quality_mean > 0.8:
                self.attachment_styles.data[0] *= 1.01
            elif quality_mean < 0.3:
                self.attachment_styles.data[3] *= 1.01
            self.attachment_styles.data = torch.softmax(self.attachment_styles.data, dim=0)

class DefenseMechanisms(nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        self.device = device
        # Update the input dimension to 398 to match the combined input
        self.mechanism_strength = nn.Sequential(
            nn.Linear(398, 256),  # Changed from 393 to 398
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU()
        ).to(device)
        self.mechanisms = nn.ModuleDict({
            'repression': nn.Linear(128, 1),
            'projection': nn.Linear(128, 1),
            'denial': nn.Linear(128, 1),
            'sublimation': nn.Linear(128, 1),
            'rationalization': nn.Linear(128, 1),
            'displacement': nn.Linear(128, 1),
            'regression': nn.Linear(128, 1)
        }).to(device)
        self.anxiety_threshold = nn.Parameter(torch.tensor(0.7, device=device))
        
    def forward(self, emotional_input: torch.Tensor, anxiety_level: torch.Tensor):
        if anxiety_level > self.anxiety_threshold:
            mechanism_features = self.mechanism_strength(emotional_input)
            defense_activations = {name: torch.sigmoid(layer(mechanism_features)) for name, layer in self.mechanisms.items()}
            strongest_defense = max(defense_activations.items(), key=lambda x: x[1].item())
            return {
                'active_defense': strongest_defense[0],
                'defense_strength': strongest_defense[1],
                'all_mechanisms': defense_activations
            }
        return {
            'active_defense': None,
            'defense_strength': torch.tensor(0.0, device=self.device),
            'all_mechanisms': {name: torch.tensor(0.0, device=self.device) for name in self.mechanisms.keys()}
        }
    
    def update_threshold(self, stress_level: float):
        self.anxiety_threshold.data = torch.clamp(
            self.anxiety_threshold.data * (1.0 + (stress_level - 0.5) * 0.1),
            min=0.3,
            max=0.9
        )
