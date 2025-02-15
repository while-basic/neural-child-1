# text_embed.py

import requests
from typing import List, Dict, Union, Optional
from config import EMBEDDING_SERVER_URL, EMBEDDING_DIM, DEFAULT_EMBEDDING
import numpy as np

def get_embeddings(text_input: Union[str, List[str]], 
                  model: str = "text-embedding-nomic-embed-text-v1.5",
                  server_url: str = EMBEDDING_SERVER_URL) -> List[Dict]: #Using a local Embedding model in LM Studio for embedding purposes.
    try:
        payload_input = [text_input] if isinstance(text_input, str) else text_input
        
        response = requests.post(
            f"{server_url}/v1/embeddings",
            json={"model": model, "input": payload_input},
            timeout=30
        )
        if response.status_code == 200:
            return response.json().get("data", [])
        print(f"Embedding server error: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"Oops! Embedding server isn't playing nice: {e}")
    except Exception as e:
        print(f"Something went sideways: {e}")
    
    # Return default embedding with correct dimensions
    return [{'embedding': np.zeros(EMBEDDING_DIM).tolist()}]
