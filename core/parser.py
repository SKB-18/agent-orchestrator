import json
import re
from typing import Dict, Any

def parse_tool_call(text: str) -> Dict[str, Any]:
    """
    Parses a tool call from LLM text output.
    Handles raw JSON and JSON embedded in markdown blocks.
    """
    json_match = re.search(r'```(?:json)?\s*(.*?)\s*```', text, re.DOTALL)
    
    json_str = ""
    if json_match:
        json_str = json_match.group(1)
    else:
        # Fallback: Find the first '{' and the last '}'
        start_idx = text.find('{')
        end_idx = text.rfind('}')
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            json_str = text[start_idx:end_idx+1]
        else:
            raise ValueError("No valid JSON object found in output.")
            
    try:
        parsed = json.loads(json_str)
    except json.JSONDecodeError:
        raise ValueError("No valid JSON object found in output.")
        
    if "tool_name" not in parsed or "kwargs" not in parsed:
        raise ValueError("Missing required keys in tool call JSON ('tool_name' and 'kwargs').")
        
    return parsed
