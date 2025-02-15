import torch
import torch.nn as nn
import config
from attachment import AttachmentSystem
from defense_mechanisms import DefenseMechanisms
from theory_of_mind import TheoryOfMind

class CustomTransformerEncoderLayer(nn.TransformerEncoderLayer):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__(d_model, nhead, dim_feedforward, dropout)
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

class DynamicNeuralChild(nn.Module):
    def __init__(self, device=None):
        super().__init__()
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Core neural components
        self.encoder = nn.Sequential(
            nn.Linear(config.EMBEDDING_DIM, config.HIDDEN_DIM),
            nn.LayerNorm(config.HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Create custom transformer encoder layer and encoder
        encoder_layer = CustomTransformerEncoderLayer(
            d_model=config.HIDDEN_DIM,
            nhead=8,
            dim_feedforward=2048,
            dropout=0.1
        )
        self.processor = CustomTransformerEncoder(encoder_layer, num_layers=4)
        
        self.decoder = nn.Sequential(
            nn.Linear(config.HIDDEN_DIM, config.HIDDEN_DIM // 2),
            nn.LayerNorm(config.HIDDEN_DIM // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config.HIDDEN_DIM // 2, config.EMBEDDING_DIM)
        )
        
        # Psychological systems
        self.attachment = AttachmentSystem(config.HIDDEN_DIM).to(device)
        self.defense_mechanisms = DefenseMechanisms().to(device)
        self.theory_of_mind = TheoryOfMind(config.HIDDEN_DIM).to(device)
        
        # State tracking
        self.internal_state = torch.zeros(1, config.HIDDEN_DIM, device=device)
        self.attention_weights = None
        
    def forward(self, x):
        # Encode input
        encoded = self.encoder(x)
        
        # Update internal state with new input
        self.internal_state = 0.9 * self.internal_state + 0.1 * encoded
        
        # Apply psychological filters
        encoded = self.attachment(encoded)
        encoded = self.defense_mechanisms(encoded)
        encoded = self.theory_of_mind(encoded)
        
        # Process through transformer
        processed = self.processor(encoded)
        # Store attention weights from the last layer
        self.attention_weights = self.processor.get_attention_weights()
        
        # Decode output
        output = self.decoder(processed)
        
        return output
    
    def get_state(self):
        return {
            'internal_state': self.internal_state.detach(),
            'attention': self.attention_weights,
            'attachment_style': self.attachment.get_style(),
            'defense_level': self.defense_mechanisms.anxiety_threshold,
            'theory_of_mind': self.theory_of_mind.get_state()
        }
