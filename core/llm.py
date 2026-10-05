import httpx
from typing import List, Dict, Any
from core.prompt import build_system_prompt
from core.parser import parse_tool_call

class BaseLLM:
    def generate(self, prompt: str, memory: List[Dict[str, Any]]) -> Dict[str, Any]:
        raise NotImplementedError

class OpenAILikeLLM(BaseLLM):
    """
    Connects to an OpenAI-compatible /v1/chat/completions endpoint.
    This could be our local Inference Lab or an external API.
    """
    def __init__(self, base_url: str = "http://127.0.0.1:8000/v1", api_key: str = "dummy", tools: List[Any] = None):
        self.base_url = base_url
        self.api_key = api_key
        self.tools = tools or []
        self.system_prompt = build_system_prompt(self.tools)
        
    def generate(self, prompt: str, memory: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Build messages payload
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(memory)
        
        # NOTE: In a real async environment we would use httpx.AsyncClient.
        # Keeping it synchronous here for simplicity of the core loop interface.
        try:
            response = httpx.post(
                f"{self.base_url}/chat/completions",
                json={
                    "model": "local-model",
                    "messages": messages,
                    "temperature": 0.0
                },
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=30.0
            )
            response.raise_for_status()
            data = response.json()
            
            raw_text = data["choices"][0]["message"]["content"]
            
            # Decide if it's a tool call or text
            try:
                # If we successfully parse it as JSON matching our tool schema
                tool_call = parse_tool_call(raw_text)
                return {
                    "type": "tool_call",
                    "tool_name": tool_call["tool_name"],
                    "kwargs": tool_call["kwargs"]
                }
            except ValueError:
                # If it's not valid tool JSON, it's a regular text response
                return {
                    "type": "text",
                    "content": raw_text
                }
                
        except Exception as e:
            # Fallback for API errors
            return {"type": "text", "content": f"LLM Connection Error: {str(e)}"}
