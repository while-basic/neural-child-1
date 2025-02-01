import torch
import torch.nn as nn
from text_embed import get_embeddings

class EmotionalEmbedder(nn.Module):
    def __init__(self):
        super().__init__()
        self.valence_proj = nn.Linear(768, 4)  # Projects to 4 emotion dimensions
        self.valence_proj.weight.data.normal_(mean=0.0, std=0.02)
        
    def forward(self, text_input):
        embed_result = get_embeddings(text_input)
        embeddings = torch.tensor(
            [item['embedding'] for item in embed_result], 
            device='cuda',
            dtype=torch.float32
        )
        valence_arousal = torch.sigmoid(self.valence_proj(embeddings))
        return {
            'semantic_embedding': embeddings,
            'valence': valence_arousal[:, 0],
            'arousal': valence_arousal[:, 1]
        }
