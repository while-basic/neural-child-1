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
from openai import OpenAI

# Initialize OpenAI client for LM Studio
client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

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
    model: str = "unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF",  # Updated default model
    temperature: float = 0.7,
    max_tokens: int = -1,
    stream: bool = False,  # Changed default to False since we're using structured output
    server_url: str = "http://localhost:1234/v1/chat/completions",
    structured_output: bool = False
) -> Optional[Dict[str, Any]]:
    """
    Enhanced chat completion using OpenAI client with LM Studio
    """
    # Generate JSON schema if structured output requested
    if structured_output:
        # Add example format to the system prompt
        system_prompt += "\nRespond in valid JSON format like this example:\n"
        system_prompt += """
{
    "content": "That's wonderful! [HUG]",
    "emotional_context": {
        "joy": 0.8,
        "trust": 0.7,
        "fear": 0.1,
        "surprise": 0.2
    },
    "reward_score": 0.7,
    "success_metric": 0.6,
    "complexity_rating": 0.3,
    "self_critique_score": 0.5,
    "cognitive_labels": ["positive_reinforcement", "emotional_support"]
}"""

    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=temperature,
            stream=stream
        )
        
        raw_content = completion.choices[0].message.content
        
        if not raw_content:
            print("Empty response from LLM server")
            return DEFAULT_RESPONSE
        
        if structured_output:
            try:
                # Try to clean up the response if it contains extra text
                raw_content = raw_content.strip()
                # Find the first { and last }
                start = raw_content.find('{')
                end = raw_content.rfind('}') + 1
                if start >= 0 and end > start:
                    raw_content = raw_content[start:end]
                
                # Parse the JSON and validate against schema
                parsed_json = json.loads(raw_content)
                return MotherResponse(**parsed_json).dict()
            except (json.JSONDecodeError, ValueError) as e:
                print(f"Failed to parse structured LLM response: {e}")
                print(f"Raw content was: {raw_content}")
                return DEFAULT_RESPONSE
        
        # For non-structured output, just return the parsed response
        return {'text': raw_content}

    except Exception as e:
        print(f"Error in chat completion: {str(e)}")
        return DEFAULT_RESPONSE

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
