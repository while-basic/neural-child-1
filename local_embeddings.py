import torch
import torch.nn as nn
import numpy as np
from typing import List
import config

class SimpleEmbeddingService:
    def __init__(self, vocab_size=30000, embedding_dim=768):
        self.device = config.DEVICE
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        
        # Create a simple embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.to(self.device)
        
        # Initialize with xavier normal
        nn.init.xavier_normal_(self.embedding.weight)
        
        # Simple tokenization (just use character codes)
        self.char_to_ix = {chr(i): i % vocab_size for i in range(128)}
    
    def tokenize(self, text: str) -> torch.Tensor:
        """Convert text to token indices"""
        tokens = [self.char_to_ix.get(c, 0) for c in text.lower()]
        return torch.tensor(tokens, device=self.device)
    
    def get_embedding(self, text: str) -> List[float]:
        """Get embedding for a single text"""
        with torch.no_grad():
            tokens = self.tokenize(text)
            embeddings = self.embedding(tokens)
            # Mean pooling
            pooled = torch.mean(embeddings, dim=0)
            # Ensure output dimension matches config
            if pooled.shape[0] != config.EMBEDDING_DIM:
                pooled = nn.functional.interpolate(
                    pooled.unsqueeze(0).unsqueeze(0),
                    size=config.EMBEDDING_DIM,
                    mode='linear'
                ).squeeze()
            return pooled.cpu().numpy().tolist()
    
    def batch_encode(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for multiple texts"""
        with torch.no_grad():
            embeddings = []
            for text in texts:
                embedding = self.get_embedding(text)
                embeddings.append(embedding)
            return embeddings

# Initialize and export global embedding service
embedding_service = SimpleEmbeddingService()
__all__ = ['SimpleEmbeddingService', 'embedding_service']
