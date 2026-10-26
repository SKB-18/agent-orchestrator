import pytest
from core.agent import Agent
from core.registry import ToolRegistry
from core.handoff import HandoffCoordinator, Specialist
from core.debate import DebateCoordinator


class MockScriptedLLM:
    def __init__(self, responses):
        self.responses = responses
        self.idx = 0

    def generate(self, prompt: str, memory: list):
        if self.idx < len(self.responses):
            resp = self.responses[self.idx]
            self.idx += 1
            return resp
        return {"type": "text", "content": "Done"}


def test_agent_handles_tool_exception():
    registry = ToolRegistry()
    def failing_tool(val):
        raise ValueError("Invalid parameter value!")
    registry.register("failing_tool", failing_tool)

    llm = MockScriptedLLM([
        {"type": "tool_call", "tool_name": "failing_tool", "kwargs": {"val": 123}},
        {"type": "text", "content": "I encountered an error and adapted."}
    ])
    agent = Agent(llm=llm, registry=registry)
    result = agent.run("Run failing task")
    
    assert result == "I encountered an error and adapted."
    # Memory should record the error
    tool_mem = agent.memory[2]
    assert tool_mem["role"] == "tool"
    assert "Error: Invalid parameter value!" in tool_mem["content"]


def test_agent_unregistered_tool():
    registry = ToolRegistry()
    llm = MockScriptedLLM([
        {"type": "tool_call", "tool_name": "non_existent_tool", "kwargs": {}},
        {"type": "text", "content": "Recovered from missing tool."}
    ])
    agent = Agent(llm=llm, registry=registry)
    result = agent.run("Call missing tool")
    
    assert result == "Recovered from missing tool."
    assert "Tool 'non_existent_tool' not found" in agent.memory[2]["content"]


def test_agent_max_steps_reached():
    registry = ToolRegistry()
    registry.register("ping", lambda: "pong")
    # LLM always returns a tool call in an infinite loop
    class InfiniteLoopLLM:
        def generate(self, prompt, memory):
            return {"type": "tool_call", "tool_name": "ping", "kwargs": {}}

    agent = Agent(llm=InfiniteLoopLLM(), registry=registry)
    result = agent.run("Loop forever", max_steps=3)
    assert result == "Max steps reached without a final answer."
    assert len(agent.memory) == 1 + (3 * 2)  # User + 3*(Assistant call + Tool response)


def test_agent_multi_step_tool_sequence():
    registry = ToolRegistry()
    registry.register("step1", lambda x: x * 2)
    registry.register("step2", lambda y: y + 10)

    llm = MockScriptedLLM([
        {"type": "tool_call", "tool_name": "step1", "kwargs": {"x": 5}},
        {"type": "tool_call", "tool_name": "step2", "kwargs": {"y": 10}},
        {"type": "text", "content": "Final result is 20"}
    ])
    agent = Agent(llm=llm, registry=registry)
    result = agent.run("Compute pipeline")
    
    assert result == "Final result is 20"
    assert agent.memory[2]["content"] == "10"
    assert agent.memory[4]["content"] == "20"


from core.debate import DebateCoordinator


def test_debate_multiround():
    class ScriptedDebateLLM:
        def generate(self, prompt, memory=None):
            if "judge" in prompt.lower():
                return {"type": "text", "content": "Judge: Consensus is Python for rapid dev and C++ for low latency."}
            elif "proposer" in prompt.lower():
                return {"type": "text", "content": "Proposal: Python is versatile and readable."}
            elif "skeptic" in prompt.lower():
                return {"type": "text", "content": "Counter: But C++ is significantly faster."}
            return {"type": "text", "content": "Default"}

    coordinator = DebateCoordinator(
        proposer_llm=ScriptedDebateLLM(),
        skeptic_llm=ScriptedDebateLLM(),
        judge_llm=ScriptedDebateLLM()
    )
    result = coordinator.conduct_debate("Python vs C++", rounds=2)
    assert "Consensus" in result["final_verdict"]
    assert len(result["transcript"]) == 4  # 2 rounds * 2 turns



def test_handoff_no_specialists_graceful():
    coordinator = HandoffCoordinator(router_llm=MockScriptedLLM([]))
    res = coordinator.orchestrate("Do anything")
    assert res["status"] == "error"
    assert "not registered" in res["error"]

