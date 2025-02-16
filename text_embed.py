# text_embed.py

import json
from typing import List, Dict, Any, Optional
import torch
import torch.nn.functional as F
import numpy as np
import requests
from config import config, DEFAULT_EMBEDDING, EMBEDDING_SERVER_URL

def get_embeddings(text: str) -> List[Dict[str, Any]]:
    """Get text embeddings from the local embedding server"""
    try:
        # Prepare request
        payload = {
            'text': text,
            'model': 'all-MiniLM-L6-v2'  # Default model
        }
        
        # Make request to local embedding server
        response = requests.post(
            f"{EMBEDDING_SERVER_URL}/embeddings",
            json=payload,
            timeout=5
        )
        
        # Check response
        if response.status_code == 200:
            result = response.json()
            if 'data' in result and len(result['data']) > 0:
                return result['data']
        
        # Return default embedding if request fails
        print(f"Warning: Using default embedding for text: {text[:50]}...")
        return [{
            'embedding': DEFAULT_EMBEDDING,
            'text': text
        }]
        
    except Exception as e:
        print(f"Error getting embeddings: {str(e)}")
        return [{
            'embedding': DEFAULT_EMBEDDING,
            'text': text
        }]

def batch_get_embeddings(texts: List[str]) -> List[Dict[str, Any]]:
    """Get embeddings for a batch of texts"""
    try:
        # Prepare request
        payload = {
            'texts': texts,
            'model': 'all-MiniLM-L6-v2'
        }
        
        # Make request
        response = requests.post(
            f"{EMBEDDING_SERVER_URL}/batch_embeddings",
            json=payload,
            timeout=10
        )
        
        # Check response
        if response.status_code == 200:
            result = response.json()
            if 'data' in result and len(result['data']) > 0:
                return result['data']
        
        # Return default embeddings if request fails
        print(f"Warning: Using default embeddings for {len(texts)} texts")
        return [
            {'embedding': DEFAULT_EMBEDDING, 'text': text}
            for text in texts
        ]
        
    except Exception as e:
        print(f"Error getting batch embeddings: {str(e)}")
        return [
            {'embedding': DEFAULT_EMBEDDING, 'text': text}
            for text in texts
        ]

def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Calculate cosine similarity between two vectors"""
    if len(a) != len(b):
        return 0.0
    
    try:
        a_array = np.array(a)
        b_array = np.array(b)
        
        dot_product = np.dot(a_array, b_array)
        norm_a = np.linalg.norm(a_array)
        norm_b = np.linalg.norm(b_array)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
            
        return dot_product / (norm_a * norm_b)
        
    except Exception:
        return 0.0

def find_most_similar(
    query_embedding: List[float],
    embeddings: List[Dict[str, Any]],
    top_k: int = 5
) -> List[Dict[str, Any]]:
    """Find the most similar embeddings to a query embedding"""
    try:
        similarities = [
            (i, cosine_similarity(query_embedding, e['embedding']))
            for i, e in enumerate(embeddings)
        ]
        
        # Sort by similarity score
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Return top k results
        return [
            {
                'text': embeddings[i]['text'],
                'similarity': score,
                'embedding': embeddings[i]['embedding']
            }
            for i, score in similarities[:top_k]
        ]
        
    except Exception as e:
        print(f"Error finding similar embeddings: {str(e)}")
        return []
