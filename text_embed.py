# text_embed.py

import os
import requests
import torch
from typing import Optional, List, Dict, Any

# Server configuration
EMBEDDING_SERVER = 'localhost'  # Use localhost directly
EMBEDDING_PORT = '1234'
EMBEDDING_URL = f"http://{EMBEDDING_SERVER}:{EMBEDDING_PORT}/v1/embeddings"

def get_embeddings(text: str) -> Optional[List[Dict[str, Any]]]:
    """Get embeddings with proper error handling"""
    try:
        response = requests.post(
            EMBEDDING_URL,
            json={"input": text},
            timeout=10  # Add timeout
        )
        response.raise_for_status()  # Raise error for bad status
        return response.json()
    except requests.exceptions.ConnectionError as e:
        print(f"Connection error to embedding server at {EMBEDDING_URL}: {str(e)}")
        return None
    except requests.exceptions.Timeout:
        print(f"Timeout connecting to embedding server at {EMBEDDING_URL}")
        return None
    except Exception as e:
        print(f"Error getting embeddings: {str(e)}")
        return None
