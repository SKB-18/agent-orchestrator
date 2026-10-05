from typing import Any, List, Dict
from core.registry import ToolRegistry

class Agent:
    def __init__(self, llm: Any, registry: ToolRegistry):
        self.llm = llm
        self.registry = registry
        self.memory: List[Dict[str, Any]] = []
        
    def run(self, user_prompt: str, max_steps: int = 5) -> str:
        self.memory.append({"role": "user", "content": user_prompt})
        
        for step in range(max_steps):
            # 1. LLM Call
            response = self.llm.generate(prompt=user_prompt, memory=self.memory)
            
            # 2. Check if text or tool call
            if response["type"] == "text":
                self.memory.append({"role": "assistant", "content": response["content"]})
                return response["content"]
                
            elif response["type"] == "tool_call":
                tool_name = response["tool_name"]
                kwargs = response.get("kwargs", {})
                
                self.memory.append({
                    "role": "assistant", 
                    "content": None, 
                    "tool_calls": [{"name": tool_name, "arguments": kwargs}]
                })
                
                # 3. Execute Tool
                try:
                    tool_result = self.registry.execute(tool_name, **kwargs)
                except Exception as e:
                    tool_result = f"Error: {str(e)}"
                    
                # 4. Append Tool Result to memory
                self.memory.append({
                    "role": "tool",
                    "name": tool_name,
                    "content": str(tool_result)
                })
                
        return "Max steps reached without a final answer."
