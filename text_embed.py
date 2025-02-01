# text_embed.py

import requests
from typing import List, Dict, Union, Optional

def get_embeddings(text_input: Union[str, List[str]], 
                  model: str = "text-embedding-nomic-embed-text-v1.5",
                  server_url: str = "http://192.168.2.12:1234/v1/embeddings") -> List[Dict]:
    try:
        payload_input = [text_input] if isinstance(text_input, str) else text_input
        
        response = requests.post(
            server_url,
            json={"model": model, "input": payload_input},
            timeout=30
        ).json()
        
        return response.get("data", [])
        
    except requests.exceptions.RequestException as e:
        print(f"Oops! Embedding server isn't playing nice: {e}")
        return []
    except Exception as e:
        print(f"Something went sideways: {e}")
        return []