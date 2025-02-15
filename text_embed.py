# text_embed.py

import json
from typing import List, Dict, Any
from config import DEFAULT_EMBEDDING
from local_embeddings import SimpleEmbeddingService, embedding_service  # Update import

def get_embeddings(text: str) -> List[Dict[str, Any]]:
    """Get embeddings for a given text using local embedding service."""
    if not text:
        print("Warning: Empty text provided to get_embeddings")
        return [{'embedding': DEFAULT_EMBEDDING}]
    
    try:
        embedding = embedding_service.get_embedding(text)
        
        # Validate embedding dimension
        if len(embedding) != config.EMBEDDING_DIM:
            print(f"Warning: Invalid embedding dimension: {len(embedding)}, expected {config.EMBEDDING_DIM}")
            return [{'embedding': DEFAULT_EMBEDDING}]
            
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
