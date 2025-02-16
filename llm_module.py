# llm_module.py

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import json
from typing import Optional, Dict, Any
from schemas import MotherResponse
from config import CHAT_SERVER_URL, DEFAULT_RESPONSE, config, DEVICE
from utils import parse_llm_response
import torch

def create_retry_session(
    retries: int = 3,
    backoff_factor: float = 0.3,
    status_forcelist: tuple = (500, 502, 503, 504),
) -> requests.Session:
    """Create a session with retry logic"""
    session = requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session

def chat_completion(
    system_prompt: str,
    user_prompt: str,
    model: str = "qwen2.5-7b-instruct",
    temperature: float = 0.7,
    max_tokens: int = -1,
    stream: bool = True,
    server_url: str = "http://localhost:1234/v1/chat/completions", # Through LM studio with a local LLM like in this case: Qwen2.5-7b-instruct
    structured_output: bool = False
) -> Optional[Dict[str, Any]]:
    """
    Enhanced chat completion with retry logic and better error handling
    """
    # Generate JSON schema if structured output requested
    response_schema = None
    if structured_output:
        response_schema = {
            "name": MotherResponse.__name__,
            "strict": True,
            "schema": MotherResponse.model_json_schema()
        }
        stream = False  # Force disable streaming

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": stream
    }

    if response_schema:
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": response_schema
        }

    headers = {"Content-Type": "application/json"}

    try:
        session = create_retry_session()
        response = session.post(
            f"{CHAT_SERVER_URL}/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=(30, 90)  # (connect timeout, read timeout)
        )
        response.raise_for_status()
        data = response.json()
        
        if not data or "choices" not in data:
            print("Invalid response format from LLM server")
            return DEFAULT_RESPONSE
            
        raw_content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        
        if not raw_content:
            print("Empty response from LLM server")
            return DEFAULT_RESPONSE
        
        if response_schema:
            try:
                return MotherResponse(**json.loads(raw_content)).dict()
            except (json.JSONDecodeError, ValueError) as e:
                print(f"Failed to parse structured LLM response: {e}")
                return DEFAULT_RESPONSE
        return parse_llm_response(data)

    except requests.exceptions.Timeout:
        print("Request timed out. Server might be overloaded.")
        return DEFAULT_RESPONSE
    except requests.exceptions.ConnectionError:
        print("Connection error. Server might be down.")
        return DEFAULT_RESPONSE
    except requests.exceptions.RequestException as e:
        print(f"Request error: {e}")
        return DEFAULT_RESPONSE
    except Exception as e:
        print(f"Unexpected error: {e}")
        return DEFAULT_RESPONSE
    return None

def _get_default_response(structured: bool = False) -> Dict[str, Any]:
    """Get default response in case of errors."""
    if structured:
        return {
            'response_text': 'I need a moment to think.',
            'emotional_context': {
                'happiness': 0.5,
                'sadness': 0.5,
                'anger': 0.5,
                'fear': 0.5
            },
            'reward_score': 0.5,
            'complexity_rating': 0.5,
            'self_critique_score': 0.5
        }
    return {'text': 'I need a moment to think.'}

class LLMModule:
    def __init__(self):
        self.device = DEVICE
        # Initialize other necessary components
    
    def process_response(self, response_text):
        try:
            # Add better error handling for response parsing
            if not response_text or not isinstance(response_text, str):
                return {
                    'text': 'I need a moment to think.',
                    'emotional_vector': torch.tensor([0.5, 0.5, 0.5, 0.5], device=self.device)
                }
            
            # Process the response
            # Add your processing logic here
            
            return {
                'text': response_text,
                'emotional_vector': torch.tensor([0.5, 0.5, 0.5, 0.5], device=self.device)
            }
        except Exception as e:
            print(f"Error processing LLM response: {str(e)}")
            return {
                'text': 'I need a moment to think.',
                'emotional_vector': torch.tensor([0.5, 0.5, 0.5, 0.5], device=self.device)
            }
