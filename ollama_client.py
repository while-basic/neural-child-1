"""
ollama_client.py
Created: 2024-02-19
Description: Ollama API client for Neural Child system with quantum-ready processing and advanced error handling.

Author: Dr. Celaya
Project: Neural Child + Meta-Learning
"""

import requests
import json
import time
import logging
from typing import Dict, List, Optional, Union, Any
from datetime import datetime
from dataclasses import dataclass
import numpy as np

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

@dataclass
class OllamaConfig:
    """Configuration for Ollama API interactions"""
    base_url: str = "http://localhost:11434"
    max_retries: int = 5
    timeout_seconds: int = 60
    backoff_factor: float = 1.5
    stream_mode: bool = False
    default_model: str = "deepseek-r1" #do not change this
    fallback_model: str = "llama3.2" #do not change this
    temperature_range: tuple = (0.1, 1.0)
    max_tokens: int = 500
    require_health_check: bool = True
    require_fallback_responses: bool = True
    log_all_prompts: bool = True
    log_all_responses: bool = True
    enable_response_validation: bool = True
    enable_response_caching: bool = True
    cache_duration_minutes: int = 30
    enable_batch_processing: bool = True
    max_concurrent_requests: int = 3

class OllamaClient:
    """Advanced Ollama API client with processing and error handling"""
    
    def __init__(self, config: Optional[OllamaConfig] = None):
        self.config = config or OllamaConfig()
        self.session = requests.Session()
        self.response_cache: Dict[str, Dict[str, Any]] = {}
        self.last_health_check = None
        self.quantum_state = {
            'coherence': 1.0,
            'entanglement': 0.0,
            'superposition': np.zeros(8)
        }
        
        # Initialize session with retry mechanism
        retry_strategy = requests.adapters.Retry(
            total=self.config.max_retries,
            backoff_factor=self.config.backoff_factor,
            status_forcelist=[429, 500, 502, 503, 504]
        )
        adapter = requests.adapters.HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        logger.info(f"Initialized Ollama client with base URL: {self.config.base_url}")
        
    def _check_health(self) -> bool:
        """Check if Ollama service is healthy"""
        try:
            # First try the list models endpoint which should always work if Ollama is running
            response = self.session.get(
                f"{self.config.base_url}/api/tags",
                timeout=5
            )
            
            if response.status_code == 200:
                self.last_health_check = datetime.now()
                logger.info("Ollama service is healthy")
                return True
                
            # If that fails, try a simple generate request as fallback
            fallback_response = self.session.post(
                f"{self.config.base_url}/api/generate",
                json={
                    "model": self.config.default_model,
                    "prompt": "test",
                    "max_tokens": 1
                },
                timeout=5
            )
            
            is_healthy = fallback_response.status_code == 200
            self.last_health_check = datetime.now()
            
            if not is_healthy:
                logger.error(f"Ollama service check failed with status: {fallback_response.status_code}")
                if fallback_response.status_code == 404:
                    logger.error(f"Model {self.config.default_model} not found. Attempting to pull...")
                    self._pull_model(self.config.default_model)
                    
            return is_healthy
            
        except requests.exceptions.ConnectionError:
            logger.error(f"Connection error: Could not connect to Ollama at {self.config.base_url}")
            return False
        except Exception as e:
            logger.error(f"Health check error: {str(e)}")
            return False
            
    def _pull_model(self, model_name: str) -> bool:
        """Pull a model from Ollama"""
        try:
            logger.info(f"Pulling model: {model_name}")
            response = self.session.post(
                f"{self.config.base_url}/api/pull",
                json={"name": model_name},
                timeout=300  # Longer timeout for model pulling
            )
            
            if response.status_code == 200:
                logger.info(f"Successfully pulled model: {model_name}")
                return True
            else:
                logger.error(f"Failed to pull model {model_name}: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Error pulling model {model_name}: {str(e)}")
            return False
        
    def _validate_response(self, response: Dict[str, Any]) -> bool:
        """Validate response format and content"""
        required_fields = ['response', 'model']
        
        # Check for required fields
        if not all(field in response for field in required_fields):
            logger.error(f"Missing required fields in response: {response}")
            return False
            
        # Check response length
        if len(response['response']) < 1:
            logger.error("Empty response received")
            return False
            
        return True
        
    def _process_quantum_state(self, prompt: str, response: str) -> None:
        """Update quantum state based on interaction"""
        # Update coherence based on response quality
        self.quantum_state['coherence'] *= 0.99
        
        # Update entanglement based on context relevance
        context_relevance = len(set(prompt.split()) & set(response.split())) / len(set(prompt.split()))
        self.quantum_state['entanglement'] = 0.7 * self.quantum_state['entanglement'] + 0.3 * context_relevance
        
        # Update superposition state
        self.quantum_state['superposition'] = np.roll(self.quantum_state['superposition'], 1)
        self.quantum_state['superposition'][0] = context_relevance
        
    def generate_response(self, prompt: str, temperature: Optional[float] = None) -> Dict[str, Any]:
        """Generate a response from Ollama"""
        if not self._check_health():
            setup_instructions = """
            Please ensure Ollama is properly set up:
            1. Install Ollama from https://ollama.ai
            2. Start the Ollama service
            3. Pull required models:
               - Run: 'ollama pull mistral'
               - Run: 'ollama pull llama2' (as fallback)
            """
            logger.error(f"Ollama service is not available\n{setup_instructions}")
            return {"response": "*Mother is temporarily unavailable. Please ensure Ollama is running and models are installed.*"}
        
        # Log prompt if enabled
        if self.config.log_all_prompts:
            logger.debug(f"Sending prompt to Ollama: {prompt[:100]}...")
        
        # Check cache if enabled
        cache_key = f"{prompt}_{self.config.default_model}_{temperature}"
        if self.config.enable_response_caching:
            cached = self.response_cache.get(cache_key)
            if cached:
                cache_time = cached.get('timestamp')
                if (datetime.now() - cache_time).total_seconds() < self.config.cache_duration_minutes * 60:
                    logger.debug("Returning cached response")
                    return cached['response']
        
        try:
            # Prepare request
            request_data = {
                "model": self.config.default_model,
                "prompt": prompt,
                "stream": self.config.stream_mode,
                "temperature": temperature or self.config.temperature_range[0],
                "max_tokens": self.config.max_tokens
            }
            
            # Send request
            response = self.session.post(
                f"{self.config.base_url}/api/generate",
                json=request_data,
                timeout=self.config.timeout_seconds
            )
            
            # Process response
            if response.status_code == 200:
                result = response.json()
                
                # Validate response if enabled
                if self.config.enable_response_validation:
                    if not self._validate_response(result):
                        raise ValueError("Invalid response format")
                
                # Update quantum state
                self._process_quantum_state(prompt, result['response'])
                
                # Cache response if enabled
                if self.config.enable_response_caching:
                    self.response_cache[cache_key] = {
                        'response': result,
                        'timestamp': datetime.now()
                    }
                
                # Log response if enabled
                if self.config.log_all_responses:
                    logger.debug(f"Received response: {result['response'][:100]}...")
                
                return result
                
            else:
                logger.error(f"Error from Ollama API: {response.status_code}")
                if self.config.require_fallback_responses:
                    logger.info("Attempting fallback model")
                    request_data["model"] = self.config.fallback_model
                    fallback_response = self.session.post(
                        f"{self.config.base_url}/api/generate",
                        json=request_data,
                        timeout=self.config.timeout_seconds
                    )
                    if fallback_response.status_code == 200:
                        return fallback_response.json()
                raise Exception(f"Failed to get response: {response.text}")
                
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            raise
            
    def get_quantum_metrics(self) -> Dict[str, float]:
        """Get current quantum state metrics"""
        return {
            'coherence': self.quantum_state['coherence'],
            'entanglement': self.quantum_state['entanglement'],
            'superposition_stability': np.std(self.quantum_state['superposition'])
        }
        
    def reset_quantum_state(self) -> None:
        """Reset quantum state to initial values"""
        self.quantum_state = {
            'coherence': 1.0,
            'entanglement': 0.0,
            'superposition': np.zeros(8)
        } 