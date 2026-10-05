import pytest
from core.agent import Agent
from core.registry import ToolRegistry

class DummyLLM:
    """Mocks an LLM to predictably trigger tool calls or text responses."""
    def __init__(self, responses):
        self.responses = responses
        self.call_count = 0
        
    def generate(self, prompt: str, memory: list):
        if self.call_count >= len(self.responses):
            return {"type": "text", "content": "Out of mock responses"}
        
        response = self.responses[self.call_count]
        self.call_count += 1
        return response

def test_agent_initialization():
    registry = ToolRegistry()
    llm = DummyLLM([])
    agent = Agent(llm=llm, registry=registry)
    assert agent.memory == []

def test_agent_run_text_only():
    registry = ToolRegistry()
    # Mock LLM returns a simple text response
    llm = DummyLLM([{"type": "text", "content": "Hello world"}])
    agent = Agent(llm=llm, registry=registry)
    
    final_response = agent.run("Say hello")
    assert final_response == "Hello world"
    assert len(agent.memory) == 2 # User message + Assistant message
    assert agent.memory[0] == {"role": "user", "content": "Say hello"}
    assert agent.memory[1] == {"role": "assistant", "content": "Hello world"}

def test_agent_run_with_tool():
    registry = ToolRegistry()
    registry.register("get_weather", lambda location: f"Sunny in {location}")
    
    # Mock LLM returns a tool call, then a final text response incorporating the tool result
    llm = DummyLLM([
        {"type": "tool_call", "tool_name": "get_weather", "kwargs": {"location": "London"}},
        {"type": "text", "content": "The weather in London is Sunny in London."}
    ])
    
    agent = Agent(llm=llm, registry=registry)
    final_response = agent.run("What is the weather in London?")
    
    assert final_response == "The weather in London is Sunny in London."
    assert len(agent.memory) == 4
    assert agent.memory[0]["role"] == "user"
    assert agent.memory[1]["role"] == "assistant"
    assert "tool_calls" in agent.memory[1]
    assert agent.memory[2]["role"] == "tool"
    assert agent.memory[2]["content"] == "Sunny in London"
    assert agent.memory[3]["role"] == "assistant"
