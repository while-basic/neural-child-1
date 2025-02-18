# llm_module.py

import requests
import json
from typing import Optional, Dict, Any, List
from schemas import MotherResponse

def chat_completion(
    system_prompt: str,
    messages: Optional[List[Dict[str, str]]] = None,
    user_prompt: Optional[str] = None,
    model: str = "local-model",
    temperature: float = 0.8,  # Slightly more creative
    max_tokens: int = 2048,
    stream: bool = False,
    server_url: str = "http://localhost:1234/v1/chat/completions",
    structured_output: bool = False
) -> str:
    """
    Enhanced chat completion function with better error handling and retries.
    Configured for LM Studio's API format.
    """
    try:
        # Prepare messages list
        message_list = []
        
        # Add system message with emotional guidance
        if system_prompt:
            enhanced_prompt = system_prompt + "\n\nPlease respond with emotional awareness and empathy. Use *asterisks* to show emotional expressions and body language."
            message_list.append({"role": "system", "content": enhanced_prompt})
            
        # Add previous messages if provided
        if messages:
            message_list.extend(messages)
            
        # Add user prompt if provided
        if user_prompt:
            message_list.append({"role": "user", "content": user_prompt})
            
        # Ensure we have at least one message
        if not message_list:
            raise ValueError("Either messages or user_prompt must be provided")
            
        # Prepare the request
        headers = {
            "Content-Type": "application/json"
        }
        
        data = {
            "messages": message_list,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False  # Always false for LM Studio
        }
        
        # Print request for debugging
        print(f"\nSending request to LM Studio:")
        print(json.dumps(data, indent=2))
        
        # Make the request with retry logic
        for attempt in range(3):  # Try up to 3 times
            try:
                response = requests.post(
                    server_url,
                    headers=headers,
                    json=data,
                    timeout=60
                )
                response.raise_for_status()
                break  # Success, exit retry loop
            except requests.exceptions.RequestException as e:
                if attempt == 2:  # Last attempt
                    print(f"Error connecting to LM Studio after 3 attempts: {str(e)}")
                    return "I apologize, but I'm having trouble connecting to my language processing system. Could you please try again in a moment?"
                print(f"Attempt {attempt + 1} failed, retrying...")
                import time
                time.sleep(1)  # Wait a second before retrying
        
        # Parse the response
        result = response.json()
        
        # Print raw response for debugging
        print(f"\nResponse from LM Studio:")
        print(json.dumps(result, indent=2))
        
        # Extract the response text
        if "choices" in result and len(result["choices"]) > 0:
            message = result["choices"][0].get("message", {})
            return message.get("content", "").strip()
            
        return "I apologize, but I'm unable to provide a response at the moment."
        
    except Exception as e:
        print(f"Error in chat_completion: {str(e)}")
        return "I apologize, but I encountered an error while processing your request. Could you please try again?"
