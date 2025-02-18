# llm_module.py

"""
llm_module.py - LLM Integration Module
Created: 2024-03-21
Description: Handles interactions with LM Studio and other LLM backends.

Author: Dr. Celaya
Project: Neural Child + Meta-Learning
"""

import requests
import time
import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from schemas import MotherResponse

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class LMStudioConnection:
    def __init__(self, 
                 base_url: str = "http://localhost:1234",
                 max_retries: int = 3,
                 timeout: int = 30,
                 backoff_factor: float = 1.5):
        self.base_url = base_url
        self.max_retries = max_retries
        self.timeout = timeout
        self.backoff_factor = backoff_factor
        self.session = requests.Session()
        
    def chat_completion(self, 
                       messages: List[Dict[str, str]], 
                       system_prompt: Optional[str] = None,
                       temperature: float = 0.8,
                       max_tokens: int = 2048) -> str:
        """
        Send a chat completion request to LM Studio with robust error handling
        """
        if system_prompt:
            messages.insert(0, {
                "role": "system",
                "content": system_prompt
            })
            
        payload = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False
        }
        
        logger.debug(f"\nSending request to LM Studio:\n{json.dumps(payload, indent=2)}")
        
        for attempt in range(self.max_retries):
            try:
                current_timeout = self.timeout * (self.backoff_factor ** attempt)
                response = self.session.post(
                    f"{self.base_url}/v1/chat/completions",
                    json=payload,
                    timeout=current_timeout
                )
                response.raise_for_status()
                
                result = response.json()
                if "choices" in result and len(result["choices"]) > 0:
                    return result["choices"][0]["message"]["content"]
                else:
                    logger.error(f"Invalid response format: {result}")
                    continue
                    
            except requests.Timeout:
                logger.warning(f"Attempt {attempt + 1} timed out after {current_timeout}s, retrying...")
                if attempt == self.max_retries - 1:
                    return self._generate_fallback_response()
                    
            except requests.RequestException as e:
                logger.error(f"Request error on attempt {attempt + 1}: {str(e)}")
                if attempt == self.max_retries - 1:
                    return self._generate_fallback_response()
                time.sleep(attempt * 2)  # Exponential backoff
                
            except Exception as e:
                logger.error(f"Unexpected error on attempt {attempt + 1}: {str(e)}")
                if attempt == self.max_retries - 1:
                    return self._generate_fallback_response()
                time.sleep(attempt * 2)
                
        return self._generate_fallback_response()
    
    def _generate_fallback_response(self) -> str:
        """Generate a graceful fallback response when LM Studio is unavailable"""
        fallback_responses = [
            "*shows gentle concern* I'm having a bit of trouble with my thoughts right now. Could we pause for a moment?",
            "*looks thoughtful* I need a moment to gather my thoughts. Shall we try again?",
            "*smiles apologetically* My mind is a little fuzzy. Could you give me a minute to clear my head?",
            "*maintains warm presence* I'm experiencing some difficulty processing right now. Let's take a brief pause."
        ]
        return fallback_responses[int(time.time()) % len(fallback_responses)]
    
    def health_check(self) -> bool:
        """Check if LM Studio is available and responding"""
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except:
            return False
            
    def __del__(self):
        """Clean up resources"""
        self.session.close()

# Global LM Studio connection instance
lm_studio = LMStudioConnection()

def chat_completion(system_prompt: str, messages: List[Dict[str, str]]) -> str:
    """
    Wrapper function for chat completion with automatic reconnection
    """
    global lm_studio
    
    # Check connection health and reinitialize if needed
    if not lm_studio.health_check():
        logger.info("Reinitializing LM Studio connection...")
        lm_studio = LMStudioConnection()
    
    return lm_studio.chat_completion(messages, system_prompt)
