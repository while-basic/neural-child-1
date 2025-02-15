import torch
import torch.nn as nn
import torch.nn.functional as F

class AttachmentSystem(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Attachment style weightings (secure, anxious, avoidant, disorganized)
        self.attachment_styles = nn.Parameter(
            torch.tensor([0.7, 0.1, 0.1, 0.1])
        )
        
        # Neural networks for each attachment component
        self.trust_network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim)
        )
        
        self.emotion_network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim)
        )
        
        self.relationship_memory = nn.GRUCell(hidden_dim, hidden_dim)
        self.hidden_state = None
        
    def forward(self, x):
        batch_size = x.size(0)
        if self.hidden_state is None or self.hidden_state.size(0) != batch_size:
            self.hidden_state = torch.zeros(batch_size, self.hidden_dim, device=x.device)
        
        # Process through attachment style filters
        trust_output = self.trust_network(x)
        emotion_output = self.emotion_network(x)
        
        # Update relationship memory
        self.hidden_state = self.relationship_memory(x, self.hidden_state)
        
        # Apply attachment style weightings
        secure_output = x * self.attachment_styles[0]
        anxious_output = trust_output * self.attachment_styles[1]
        avoidant_output = emotion_output * self.attachment_styles[2]
        disorganized_output = self.hidden_state * self.attachment_styles[3]
        
        # Combine outputs based on attachment styles
        combined_output = (
            secure_output +
            anxious_output +
            avoidant_output +
            disorganized_output
        )
        
        return combined_output
    
    def get_style(self):
        return {
            'secure': self.attachment_styles[0].item(),
            'anxious': self.attachment_styles[1].item(),
            'avoidant': self.attachment_styles[2].item(),
            'disorganized': self.attachment_styles[3].item()
        }
        
    def update_style(self, experience_tensor):
        """Update attachment styles based on experiences"""
        with torch.no_grad():
            # Gradual adjustment based on positive/negative experiences
            delta = torch.tanh(experience_tensor) * 0.01
            self.attachment_styles.data += delta
            self.attachment_styles.data = F.softmax(self.attachment_styles.data, dim=0)
