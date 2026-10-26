import pytest
from fastapi.testclient import TestClient
from server.app import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "specialists" in data
    assert "math" in data["specialists"]
    assert "search" in data["specialists"]
    assert "code" in data["specialists"]

def test_chat_endpoint():
    response = client.post("/v1/agent/chat", json={
        "message": "What is 2 + 2?",
        "max_steps": 2
    })
    assert response.status_code == 200
    data = response.json()
    assert "final_response" in data
    assert "memory_trace" in data

def test_handoff_endpoint():
    response = client.post("/v1/agent/handoff", json={
        "query": "Calculate 15 plus 25",
        "max_steps": 2
    })
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["routed_specialist"] == "math"
    assert "final_answer" in data
    assert "execution_trace" in data

def test_reflect_endpoint():
    response = client.post("/v1/agent/reflect", json={
        "problem": "A train travels 60 miles in 1 hour. How far in 3 hours?"
    })
    assert response.status_code == 200
    data = response.json()
    assert "problem" in data
    assert "draft" in data
    assert "critique" in data
    assert "final_solution" in data

def test_debate_endpoint():
    response = client.post("/v1/agent/debate", json={
        "topic": "Python vs Rust for backend performance",
        "rounds": 1
    })
    assert response.status_code == 200
    data = response.json()
    assert "topic" in data
    assert "transcript" in data
    assert "final_verdict" in data

def test_invalid_request_validation():
    response = client.post("/v1/agent/chat", json={})
    assert response.status_code == 422
