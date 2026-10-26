import pytest
from core.reflection import ReflectionAgent
from core.llm import MockLLM


def test_reflection_agent_three_phase_flow():
    # Script generator and critic responses
    generator_llm = MockLLM([
        {"type": "text", "content": "Draft: 15 apples minus 3 equals 12 apples."},
        {"type": "text", "content": "Refinement: Incorporating review, the final answer is 12 apples."}
    ])
    critic_llm = MockLLM([
        {"type": "text", "content": "Critique: Step 1 is correct. Calculation 15 - 3 = 12 holds."}
    ])

    reflection_agent = ReflectionAgent(generator_llm=generator_llm, critic_llm=critic_llm)
    problem = "John has 15 apples and eats 3. How many are left?"
    
    result = reflection_agent.solve(problem=problem)
    
    assert result["problem"] == problem
    assert "Draft:" in result["draft"]
    assert "Critique:" in result["critique"]
    assert "Refinement:" in result["final_solution"]
    assert len(result["trace"]) == 3
    assert result["trace"][0]["phase"] == "draft"
    assert result["trace"][1]["phase"] == "critique"
    assert result["trace"][2]["phase"] == "refinement"
