# llm_module.py

import requests
import json
from typing import Optional, Dict, Any
from schemas import MotherResponse

def chat_completion(
    system_prompt: str,
    user_prompt: str,
    model: str = "qwen2.5-7b-instruct",
    temperature: float = 0.7,
    max_tokens: int = -1,
    stream: bool = True,
    server_url: str = "http://192.168.2.12:1234/v1/chat/completions",
    structured_output: bool = False
) -> Optional[Dict[str, Any]]:
    """
    Enhanced with automatic structured response handling
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
        if stream:
            collected_content = []
            with requests.post(server_url, headers=headers, json=payload, stream=True, timeout=60) as response:
                response.raise_for_status()

                for chunk in response.iter_lines(decode_unicode=True):
                    if chunk:
                        try:
                            data = json.loads(chunk[6:])  # Remove "data: " prefix
                            delta = data.get("choices", [{}])[0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                collected_content.append(content)
                                print(content, end="", flush=True)
                        except json.JSONDecodeError:
                            continue

                print()
                return "".join(collected_content)

        else:
            response = requests.post(server_url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            data = response.json()
            raw_content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            
            if response_schema:
                try:
                    return MotherResponse(**json.loads(raw_content)).dict()
                except json.JSONDecodeError:
                    print("Failed to parse structured LLM response")
                    return None
            return raw_content

    except requests.exceptions.RequestException as e:
        print(f"Request error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
    return None