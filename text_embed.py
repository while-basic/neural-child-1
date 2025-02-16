# text_embed.py

import json
from typing import List, Dict, Any
import torch
import torch.nn.functional as F
from config import config, DEFAULT_EMBEDDING
from local_embeddings import SimpleEmbeddingService, embedding_service

def get_embeddings(text: str) -> List[Dict[str, Any]]:
    """Get embeddings for a given text using local embedding service."""
    if not text:
        print("Warning: Empty text provided to get_embeddings")
        return [{'embedding': DEFAULT_EMBEDDING}]
    
    try:
        embedding = embedding_service.get_embedding(text)
        
        # Convert to tensor for dimension handling
        embedding_tensor = torch.tensor(embedding)
        
        # Ensure correct dimension
        if embedding_tensor.size(0) != embedding_service.embedding_dim:
            if embedding_tensor.size(0) > embedding_service.embedding_dim:
                embedding_tensor = embedding_tensor[:embedding_service.embedding_dim]
            else:
                embedding_tensor = F.pad(embedding_tensor, (0, embedding_service.embedding_dim - embedding_tensor.size(0)))
        
        # Convert back to list for return
        embedding = embedding_tensor.tolist()
            
        return [{'embedding': embedding}]
    except Exception as e:
        print(f"Embedding error: {e}")
        return [{'embedding': DEFAULT_EMBEDDING}]

def batch_get_embeddings(texts: List[str]) -> List[Dict[str, Any]]:
    """Get embeddings for multiple texts."""
    try:
        embeddings = embedding_service.batch_encode(texts)
        return [{'embedding': emb} for emb in embeddings]
    except Exception as e:
        print(f"Batch embedding error: {e}")
        return [{'embedding': DEFAULT_EMBEDDING} for _ in texts]
