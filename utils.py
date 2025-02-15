import json
from config import DEFAULT_RESPONSE

def parse_llm_response(response_text):
    """Parse LLM response and ensure it matches expected format"""
    try:
        if isinstance(response_text, dict):
            return response_text
        # Try to find JSON in the response
        start_idx = response_text.find('{')
        end_idx = response_text.rfind('}') + 1
        if start_idx != -1 and end_idx != 0:
            json_str = response_text[start_idx:end_idx]
            return json.loads(json_str)
    except (json.JSONDecodeError, AttributeError, TypeError) as e:
        print(f"Failed to parse LLM response: {e}")
    return DEFAULT_RESPONSE
