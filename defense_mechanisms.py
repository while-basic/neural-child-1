import torch
import torch.nn as nn
import torch.nn.functional as F

class DefenseMechanisms(nn.Module):
    def __init__(self):
        super().__init__()
        # Anxiety threshold for triggering defenses
        self.anxiety_threshold = nn.Parameter(torch.tensor(0.7))
        
        # Defense mechanism strengths
        self.defense_strengths = nn.Parameter(torch.ones(5) / 5)  # 5 defense mechanisms
        
    def forward(self, x):
        # Calculate anxiety level from input
        anxiety_level = torch.mean(torch.abs(x))
        
        if anxiety_level > self.anxiety_threshold:
            # Apply defense mechanisms
            x = self._apply_defenses(x)
            
        return x
    
    def _apply_defenses(self, x):
        # Denial: suppress extreme values
        x = x * (1 - self.defense_strengths[0])
        
        # Projection: reverse negative values
        x = torch.where(x < 0, -x * self.defense_strengths[1], x)
        
        # Rationalization: smooth out variations
        x = F.avg_pool1d(x.unsqueeze(1), kernel_size=3, stride=1, padding=1).squeeze(1)
        x = x * self.defense_strengths[2]
        
        # Displacement: shift emotional content
        x = torch.roll(x, shifts=1, dims=-1) * self.defense_strengths[3]
        
        # Sublimation: transform negative to positive
        x = torch.where(x < 0, torch.abs(x) * self.defense_strengths[4], x)
        
        return x
