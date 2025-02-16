import json
import torch
from typing import Dict, Any, Optional, Union
from config import DEFAULT_RESPONSE, config
import numpy as np
from datetime import datetime

def parse_llm_response(response_text: str) -> Dict[str, Any]:
    """Parse LLM response and ensure it matches expected format"""
    try:
        if isinstance(response_text, dict):
            return response_text
            
        # Try to find JSON in the response
        start_idx = response_text.find('{')
        end_idx = response_text.rfind('}') + 1
        
        if start_idx != -1 and end_idx != 0:
            json_str = response_text[start_idx:end_idx]
            try:
                parsed = json.loads(json_str)
                # Ensure required fields exist
                if 'content' not in parsed:
                    parsed['content'] = DEFAULT_RESPONSE['text']
                if 'emotional_context' not in parsed:
                    parsed['emotional_context'] = DEFAULT_RESPONSE['emotional_context']
                return parsed
            except json.JSONDecodeError:
                print(f"Failed to parse JSON: {json_str}")
                return DEFAULT_RESPONSE
    except Exception as e:
        print(f"Error parsing LLM response: {e}")
        return DEFAULT_RESPONSE

def ensure_tensor(data: Union[list, np.ndarray, torch.Tensor], device: Optional[torch.device] = None) -> torch.Tensor:
    """Convert input data to tensor and ensure it's on the correct device"""
    if device is None:
        device = config.DEVICE
        
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data).to(device)
    elif isinstance(data, list):
        return torch.tensor(data, device=device)
    else:
        raise ValueError(f"Unsupported data type: {type(data)}")

def format_time_delta(start_time: datetime) -> str:
    """Format time delta in a human-readable format"""
    delta = datetime.now() - start_time
    days = delta.days
    hours = delta.seconds // 3600
    minutes = (delta.seconds % 3600) // 60
    seconds = delta.seconds % 60
    
    parts = []
    if days > 0:
        parts.append(f"{days}d")
    if hours > 0 or days > 0:
        parts.append(f"{hours}h")
    if minutes > 0 or hours > 0 or days > 0:
        parts.append(f"{minutes}m")
    parts.append(f"{seconds}s")
    
    return " ".join(parts)

def calculate_moving_average(values: list, window: int = 10) -> list:
    """Calculate moving average with specified window size"""
    if not values:
        return []
    if len(values) < window:
        return values
        
    result = []
    for i in range(len(values)):
        start_idx = max(0, i - window + 1)
        window_values = values[start_idx:i + 1]
        result.append(sum(window_values) / len(window_values))
    return result

def extract_action_marker(text: str) -> Optional[str]:
    """Extract action marker from text enclosed in [brackets]"""
    try:
        start = text.find('[')
        end = text.find(']')
        if start != -1 and end != -1 and end > start:
            return text[start + 1:end].strip()
        return None
    except Exception:
        return None

def validate_emotional_vector(vector: Union[list, np.ndarray, torch.Tensor]) -> bool:
    """Validate that emotional vector values are within valid range"""
    try:
        if isinstance(vector, torch.Tensor):
            vector = vector.cpu().numpy()
        elif isinstance(vector, list):
            vector = np.array(vector)
            
        return (
            len(vector) == config.emotion_dim and
            np.all(vector >= 0) and
            np.all(vector <= 1)
        )
    except Exception:
        return False

def create_error_response(error_type: str, details: str) -> Dict[str, Any]:
    """Create a standardized error response"""
    return {
        'error': True,
        'error_type': error_type,
        'details': details,
        'timestamp': datetime.now().isoformat(),
        'response': DEFAULT_RESPONSE
    }
