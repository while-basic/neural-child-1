# llm_module.py

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import json
from typing import Optional, Dict, Any, List
from schemas import MotherResponse, EmotionalContext, ActionType
from config import CHAT_SERVER_URL, DEFAULT_RESPONSE, config, DEVICE
from utils import parse_llm_response, ensure_tensor, create_error_response, validate_emotional_vector
import torch
from openai import OpenAI
import time
from datetime import datetime

# Initialize OpenAI client for LM Studio
client = OpenAI(
    base_url=CHAT_SERVER_URL,
    api_key="not-needed"
)

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
    model: str = config.DEFAULT_MODEL,
    temperature: float = config.temperature,
    max_tokens: int = config.max_response_tokens,
    stream: bool = False,
    structured_output: bool = False
) -> Optional[Dict[str, Any]]:
    """
    Enhanced chat completion using OpenAI client with LM Studio
    """
    # Generate JSON schema if structured output requested
    if structured_output:
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
    "action": "COMFORT",
    "reward_score": 0.7,
    "success_metric": 0.6,
    "complexity_rating": 0.3,
    "self_critique_score": 0.5,
    "cognitive_labels": ["positive_reinforcement", "emotional_support"],
    "effectiveness": 0.75,
    "developmental_focus": {
        "emotional_regulation": 0.8,
        "social_skills": 0.6,
        "cognitive_development": 0.4
    }
}"""

    try:
        start_time = time.time()
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream
        )
        
        if stream:
            return completion  # Return the stream directly
            
        raw_content = completion.choices[0].message.content
        response_time = time.time() - start_time
        
        if not raw_content:
            return create_error_response(
                "empty_response",
                "Empty response from LLM server"
            )
        
        if structured_output:
            try:
                # Clean up the response
                raw_content = raw_content.strip()
                start = raw_content.find('{')
                end = raw_content.rfind('}') + 1
                if start >= 0 and end > start:
                    raw_content = raw_content[start:end]
                
                # Parse and validate the response
                parsed_json = json.loads(raw_content)
                response = MotherResponse(**parsed_json)
                
                # Add metadata
                result = response.dict()
                result['metadata'] = {
                    'response_time': response_time,
                    'timestamp': datetime.now().isoformat(),
                    'model': model
                }
                return result
                
            except (json.JSONDecodeError, ValueError) as e:
                return create_error_response(
                    "parse_error",
                    f"Failed to parse structured response: {str(e)}"
                )
        
        # For non-structured output, return the parsed response
        return {
            'text': raw_content,
            'metadata': {
                'response_time': response_time,
                'timestamp': datetime.now().isoformat(),
                'model': model
            }
        }

    except Exception as e:
        return create_error_response(
            "llm_error",
            f"Error in chat completion: {str(e)}"
        )

class LLMModule:
    def __init__(self):
        self.device = DEVICE
        self.session = create_retry_session()
        self.response_cache = {}
        self.error_count = 0
        self.last_error_time = None
    
    def process_response(self, response_text: str) -> Dict[str, Any]:
        """Process LLM response with enhanced error handling and caching"""
        try:
            # Check cache first
            cache_key = hash(response_text)
            if cache_key in self.response_cache:
                return self.response_cache[cache_key]
            
            # Basic validation
            if not response_text or not isinstance(response_text, str):
                self._log_error("invalid_input", "Invalid response text")
                return DEFAULT_RESPONSE
            
            # Parse the response
            parsed = parse_llm_response(response_text)
            
            # Extract emotional vector
            emotional_context = parsed.get('emotional_context', {})
            emotional_vector = [
                emotional_context.get('joy', 0.5),
                emotional_context.get('trust', 0.5),
                emotional_context.get('fear', 0.1),
                emotional_context.get('surprise', 0.3)
            ]
            
            # Validate emotional vector
            if not validate_emotional_vector(emotional_vector):
                self._log_error("invalid_emotion", "Invalid emotional vector")
                emotional_vector = [0.5, 0.5, 0.1, 0.3]
            
            # Create the processed response
            processed_response = {
                'text': parsed.get('content', DEFAULT_RESPONSE['text']),
                'emotional_vector': ensure_tensor(emotional_vector, self.device),
                'action': parsed.get('action'),
                'effectiveness': float(parsed.get('effectiveness', 0.5)),
                'complexity': float(parsed.get('complexity_rating', 0.3))
            }
            
            # Cache the result
            self.response_cache[cache_key] = processed_response
            
            return processed_response
            
        except Exception as e:
            self._log_error("processing_error", str(e))
            return {
                'text': DEFAULT_RESPONSE['text'],
                'emotional_vector': ensure_tensor([0.5, 0.5, 0.5, 0.5], self.device),
                'action': None,
                'effectiveness': 0.5,
                'complexity': 0.3
            }
    
    def _log_error(self, error_type: str, details: str):
        """Log error and update error tracking"""
        self.error_count += 1
        self.last_error_time = datetime.now()
        print(f"LLM Error ({error_type}): {details}")
        
        # Reset error count periodically
        if self.error_count > 10:
            time_since_first_error = (datetime.now() - self.last_error_time).total_seconds()
            if time_since_first_error > 300:  # Reset after 5 minutes
                self.error_count = 0
