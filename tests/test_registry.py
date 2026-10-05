import pytest
from core.registry import ToolRegistry

def test_tool_registry_registration():
    registry = ToolRegistry()
    
    def dummy_tool(x: int) -> int:
        """Returns x"""
        return x
        
    registry.register("dummy_tool", dummy_tool)
    
    assert "dummy_tool" in registry.tools
    assert registry.tools["dummy_tool"] == dummy_tool
    
def test_tool_registry_execution():
    registry = ToolRegistry()
    
    def add(a: int, b: int) -> int:
        return a + b
        
    registry.register("add", add)
    
    result = registry.execute("add", a=5, b=3)
    assert result == 8
    
def test_tool_registry_missing_tool():
    registry = ToolRegistry()
    with pytest.raises(ValueError, match="Tool 'nonexistent' not found"):
        registry.execute("nonexistent")
