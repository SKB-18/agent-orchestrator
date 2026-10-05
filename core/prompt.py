from typing import List, Dict, Any
import inspect

def generate_tool_schema(func) -> Dict[str, Any]:
    """
    Generates a JSON schema for a python function.
    Useful for injecting into the LLM system prompt.
    """
    sig = inspect.signature(func)
    params = {}
    for name, param in sig.parameters.items():
        if name == 'self':
            continue
        params[name] = {
            "type": "string" if param.annotation == inspect.Parameter.empty else str(param.annotation.__name__),
        }
        
    return {
        "name": func.__name__,
        "description": func.__doc__ or f"Executes {func.__name__}",
        "parameters": params
    }

def build_system_prompt(tools: List[Any]) -> str:
    """
    Builds the system prompt containing available tools.
    """
    prompt = "You are a helpful AI assistant with access to tools.\n"
    prompt += "When you need to use a tool, you MUST output a JSON object in the following format enclosed in markdown ```json ... ``` blocks:\n"
    prompt += '{\n  "tool_name": "name_of_tool",\n  "kwargs": {"arg1": "value1"}\n}\n\n'
    
    if tools:
        prompt += "Available Tools:\n"
        for tool in tools:
            schema = generate_tool_schema(tool)
            prompt += f"- {schema['name']}: {schema['description']}\n"
            prompt += f"  Parameters: {schema['parameters']}\n"
            
    return prompt
