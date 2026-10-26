import pytest
from core.handoff import Specialist, HandoffCoordinator
from core.registry import ToolRegistry
from core.llm import MockLLM


def test_specialist_initialization_and_execution():
    registry = ToolRegistry()
    registry.register("multiply", lambda a, b: a * b)
    
    # Mock LLM calls the tool and then emits final text
    mock_llm = MockLLM([
        {"type": "tool_call", "tool_name": "multiply", "kwargs": {"a": 6, "b": 7}},
        {"type": "text", "content": "42"}
    ])
    
    specialist = Specialist("MathSpecialist", "Computes arithmetic", mock_llm, registry)
    result = specialist.execute("Calculate 6 * 7")
    
    assert result["specialist"] == "MathSpecialist"
    assert result["response"] == "42"
    assert len(result["memory_trace"]) > 1


def test_handoff_router_keyword_classification():
    router_llm = MockLLM([])
    coordinator = HandoffCoordinator(router_llm)
    
    dummy_reg = ToolRegistry()
    mock_worker = MockLLM([{"type": "text", "content": "Done"}])
    
    coordinator.register_specialist(Specialist("math", "Math tools", mock_worker, dummy_reg))
    coordinator.register_specialist(Specialist("code", "Code tools", mock_worker, dummy_reg))
    coordinator.register_specialist(Specialist("search", "Search tools", mock_worker, dummy_reg))
    
    assert coordinator.route_query("Please calculate 10 + 5") == "math"
    assert coordinator.route_query("Write a python script") == "code"
    assert coordinator.route_query("Who is Albert Einstein?") == "search"


def test_handoff_coordinator_orchestration_flow():
    router_llm = MockLLM([{"type": "text", "content": "math"}])
    coordinator = HandoffCoordinator(router_llm)
    
    reg = ToolRegistry()
    reg.register("add", lambda a, b: a + b)
    specialist_llm = MockLLM([
        {"type": "tool_call", "tool_name": "add", "kwargs": {"a": 20, "b": 30}},
        {"type": "text", "content": "Result is 50"}
    ])
    
    coordinator.register_specialist(Specialist("math", "Math agent", specialist_llm, reg))
    
    res = coordinator.orchestrate("Add 20 and 30")
    assert res["status"] == "success"
    assert res["routed_specialist"] == "math"
    assert res["final_answer"] == "Result is 50"
    assert len(res["execution_trace"]) >= 2
