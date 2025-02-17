import torch
import torch.nn as nn
from config import config
from attachment import AttachmentSystem
from defense_mechanisms import DefenseMechanisms
from theory_of_mind import TheoryOfMind
from typing import Dict, Any, Optional

class CustomTransformerEncoderLayer(nn.TransformerEncoderLayer):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.stored_attention_weights = None

    def _sa_block(self, x, attn_mask, key_padding_mask, is_causal=False):
        """Modified self-attention block to store attention weights"""
        x, attention_weights = self.self_attn(
            x, x, x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            is_causal=is_causal,
            need_weights=True
        )
        self.stored_attention_weights = attention_weights
        return self.dropout1(x)

class CustomTransformerEncoder(nn.TransformerEncoder):
    def __init__(self, encoder_layer, num_layers):
        super().__init__(encoder_layer, num_layers)
    
    def get_attention_weights(self):
        """Retrieve attention weights from the last layer"""
        return self.layers[-1].stored_attention_weights

class AttachmentSystem(nn.Module):
    def __init__(self, input_dim: int = 384, hidden_dim: int = 256):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Add input projection for dimension mismatch
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
        
        self.attachment_network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 4)  # 4 attachment styles
        )
        self.attachment_styles = nn.Parameter(torch.ones(4) / 4)  # Initialize with equal weights
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Handle input dimension
        if x.size(-1) != self.hidden_dim:
            x = self.input_projection(x)
            
        attachment_scores = self.attachment_network(x)
        attachment_weights = torch.softmax(attachment_scores, dim=-1)
        self.attachment_styles.data = 0.95 * self.attachment_styles.data + 0.05 * attachment_weights.mean(0)
        return {
            'attachment_scores': attachment_scores,
            'current_style': attachment_weights
        }

class DefenseMechanisms(nn.Module):
    def __init__(self, input_dim: int = 384, hidden_dim: int = 256):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Add input projection for dimension mismatch
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
        
        self.anxiety_threshold = nn.Parameter(torch.tensor(0.5))
        self.defense_network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim)
        )
        
    def forward(self, x: torch.Tensor, anxiety_level: float) -> torch.Tensor:
        # Handle input dimension
        if x.size(-1) != self.hidden_dim:
            x = self.input_projection(x)
            
        if anxiety_level > self.anxiety_threshold:
            return self.defense_network(x)
        return x

class TheoryOfMind(nn.Module):
    def __init__(self, input_dim: int = 384, hidden_dim: int = 256):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Add input projection for dimension mismatch
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
        
        self.perspective_network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.social_bias = nn.Parameter(torch.zeros(hidden_dim))
        
        # Add output projection to match input dimension
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, input_dim),
            nn.LayerNorm(input_dim)
        )
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Handle input dimension
        if x.size(-1) != self.hidden_dim:
            x = self.input_projection(x)
            
        perspective = self.perspective_network(x)
        social_context = perspective + self.social_bias
        
        # Project back to input dimension if needed
        if self.input_dim != self.hidden_dim:
            perspective = self.output_projection(perspective)
            social_context = self.output_projection(social_context)
            
        return {
            'perspective': perspective,
            'social_context': social_context
        }

class DynamicNeuralChild(nn.Module):
    def __init__(self, 
                 input_dim: int = config.embedding_dim,
                 hidden_dim: int = config.hidden_dim,
                 device: Optional[torch.device] = None):
        super().__init__()
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Input projection layer to handle dimension mismatch
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        # Core neural architecture
        self.encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.processor = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim)  # Project back to input dimension
        )
        
        # Psychological components
        self.attachment = AttachmentSystem(input_dim, hidden_dim)
        self.defense_mechanisms = DefenseMechanisms(input_dim, hidden_dim)
        self.theory_of_mind = TheoryOfMind(input_dim, hidden_dim)
        
        # Developmental parameters
        self.developmental_scale = nn.Parameter(torch.tensor(0.1))  # Starts at 0.1, grows with development
        self.cognitive_complexity = nn.Parameter(torch.tensor(0.1))
        self.emotional_granularity = nn.Parameter(torch.tensor(0.1))
        
        # Move to device
        self.to(self.device)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure input is on the correct device
        x = x.to(self.device)
        
        # Handle input dimension
        if x.size(-1) != self.hidden_dim:
            x = self.input_projection(x)
        
        # Basic encoding
        encoded = self.encoder(x)
        
        # Apply attachment system
        attachment_info = self.attachment(encoded)
        encoded = encoded * (1 + attachment_info['current_style'].unsqueeze(1))
        
        # Process through transformer with developmental scaling
        batch_size = encoded.size(0)
        sequence_length = max(1, int(16 * self.developmental_scale.item()))
        
        # Repeat the encoding to create a sequence
        sequence = encoded.unsqueeze(1).repeat(1, sequence_length, 1)
        
        # Apply transformer processing
        processed = self.processor(sequence)
        
        # Pool the sequence
        processed = torch.mean(processed, dim=1)
        
        # Apply defense mechanisms based on current anxiety
        anxiety = torch.mean(attachment_info['current_style'][:, 3])  # Use anxious attachment as proxy
        processed = self.defense_mechanisms(processed, anxiety.item())
        
        # Apply theory of mind
        social_processing = self.theory_of_mind(processed)
        processed = processed + self.cognitive_complexity * social_processing['social_context']
        
        # Decode with emotional granularity
        output = self.decoder(processed)
        output = output * (1 + self.emotional_granularity)
        
        return output
    
    def update_developmental_parameters(self, stage_progress: float):
        """Update developmental parameters based on stage progress"""
        with torch.no_grad():
            # Scale from 0.1 to 1.0 based on progress
            self.developmental_scale.data = torch.tensor(0.1 + 0.9 * stage_progress)
            self.cognitive_complexity.data = torch.tensor(0.1 + 0.9 * stage_progress)
            self.emotional_granularity.data = torch.tensor(0.1 + 0.9 * stage_progress)
    
    def get_developmental_metrics(self) -> Dict[str, float]:
        """Get current developmental metrics"""
        return {
            'developmental_scale': self.developmental_scale.item(),
            'cognitive_complexity': self.cognitive_complexity.item(),
            'emotional_granularity': self.emotional_granularity.item(),
            'attachment_styles': self.attachment.attachment_styles.tolist(),
            'anxiety_threshold': self.defense_mechanisms.anxiety_threshold.item(),
            'social_bias': torch.mean(self.theory_of_mind.social_bias).item()
        }

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor], strict: bool = True):
        """Custom state dict loading to handle architecture changes"""
        try:
            # Try loading with strict mode first
            super().load_state_dict(state_dict, strict=True)
        except Exception as e:
            print(f"Warning: Strict loading failed ({str(e)}), attempting flexible loading...")
            
            # Create new state dict with only matching keys
            model_state = self.state_dict()
            flexible_state = {}
            
            for key in model_state.keys():
                if key in state_dict:
                    # Check if shapes match
                    if state_dict[key].shape == model_state[key].shape:
                        flexible_state[key] = state_dict[key]
                    else:
                        print(f"Shape mismatch for {key}: expected {model_state[key].shape}, got {state_dict[key].shape}")
            
            # Load only matching parameters
            super().load_state_dict(flexible_state, strict=False)
            print("Flexible loading completed with partial state restoration")
