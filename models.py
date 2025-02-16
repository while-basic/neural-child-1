import torch
import torch.nn as nn

class ChildBrain(nn.Module):
    def __init__(self):
        super().__init__()
        # ...existing code...
        
        self.transformer = nn.Transformer(
            d_model=hidden_size,
            nhead=8,
            num_encoder_layers=6,
            num_decoder_layers=6,
            dim_feedforward=2048,
            batch_first=True  # Add this parameter
        )
        
        # ...existing code...
