import pytest
from core.debate import DebateCoordinator
from core.llm import MockLLM


def test_debate_coordinator_multi_round():
    proposer_llm = MockLLM([
        {"type": "text", "content": "Round 1 Proposer: Microservices provide isolation and scalability."},
        {"type": "text", "content": "Round 2 Proposer: Operational complexity can be managed with Kubernetes."}
    ])
    skeptic_llm = MockLLM([
        {"type": "text", "content": "Round 1 Skeptic: Monoliths are simpler to deploy and debug."},
        {"type": "text", "content": "Round 2 Skeptic: Network latency and distributed tracing overhead remain high."}
    ])
    judge_llm = MockLLM([
        {"type": "text", "content": "Final Ruling: Both paradigms have merit. Choose based on team scale."}
    ])

    coordinator = DebateCoordinator(
        proposer_llm=proposer_llm,
        skeptic_llm=skeptic_llm,
        judge_llm=judge_llm
    )
    
    topic = "Should early-stage startups use Microservices or Monoliths?"
    result = coordinator.conduct_debate(topic=topic, rounds=2)
    
    assert result["topic"] == topic
    assert result["rounds_conducted"] == 2
    assert len(result["transcript"]) == 4  # 2 rounds * 2 participants
    assert result["transcript"][0]["role"] == "Proposer"
    assert result["transcript"][1]["role"] == "Skeptic"
    assert "Final Ruling:" in result["final_verdict"]
