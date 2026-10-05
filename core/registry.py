from typing import Callable, Dict, Any

class ToolRegistry:
    def __init__(self):
        self.tools: Dict[str, Callable] = {}
        
    def register(self, name: str, func: Callable):
        self.tools[name] = func
        
    def execute(self, name: str, **kwargs) -> Any:
        if name not in self.tools:
            raise ValueError(f"Tool '{name}' not found.")
        return self.tools[name](**kwargs)
