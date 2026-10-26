import httpx
from typing import List, Dict, Any, Optional
from core.prompt import build_system_prompt
from core.parser import parse_tool_call


class BaseLLM:
    def generate(self, prompt: str, memory: List[Dict[str, Any]]) -> Dict[str, Any]:
        raise NotImplementedError


class OpenAILikeLLM(BaseLLM):
    """
    Connects to an OpenAI-compatible /v1/chat/completions endpoint.
    Targets local Inference Lab (:8002) or Ollama (:11434).
    """
    def __init__(
        self, 
        base_url: str = "http://127.0.0.1:8002/v1", 
        api_key: str = "dummy", 
        tools: List[Any] = None,
        system_prompt: Optional[str] = None
    ):
        self.base_url = base_url
        self.api_key = api_key
        self.tools = tools or []
        self.system_prompt = system_prompt or build_system_prompt(self.tools)
        
    def generate(self, prompt: str, memory: List[Dict[str, Any]]) -> Dict[str, Any]:
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(memory)
        
        try:
            response = httpx.post(
                f"{self.base_url}/chat/completions",
                json={
                    "model": "local-model",
                    "messages": messages,
                    "temperature": 0.0,
                    "max_tokens": 200
                },
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=30.0
            )
            response.raise_for_status()
            data = response.json()
            
            raw_text = data["choices"][0]["message"]["content"]
            
            # Decide if it's a tool call or direct text response
            try:
                tool_call = parse_tool_call(raw_text)
                return {
                    "type": "tool_call",
                    "tool_name": tool_call["tool_name"],
                    "kwargs": tool_call["kwargs"]
                }
            except ValueError:
                return {
                    "type": "text",
                    "content": raw_text
                }
                
        except Exception as e:
            return {"type": "text", "content": f"LLM Connection Error: {str(e)}"}


class MockLLM(BaseLLM):
    """
    Deterministic scripted LLM for testing multi-agent loops and tool pipelines.
    """
    def __init__(self, responses: Optional[List[Dict[str, Any]]] = None):
        self.responses = responses or []
        self.call_count = 0

    def generate(self, prompt: str, memory: List[Dict[str, Any]]) -> Dict[str, Any]:
        if self.call_count < len(self.responses):
            resp = self.responses[self.call_count]
            self.call_count += 1
            return resp
        return {"type": "text", "content": "Default mock response."}
