import pytest
from fastapi.testclient import TestClient
from server.app import app

client = TestClient(app)

def test_chat_endpoint_connection_error():
    # Since we don't have a real inference server running on port 8000 for this test,
    # the OpenAILikeLLM should gracefully catch the httpx ConnectionError 
    # and return a text message saying "LLM Connection Error".
    
    response = client.post("/v1/agent/chat", json={
        "message": "What time is it?",
        "max_steps": 2
    })
    
    assert response.status_code == 200
    data = response.json()
    assert "LLM Connection Error" in data["final_response"]
    assert len(data["memory_trace"]) == 2 # User prompt + Error response
    assert data["memory_trace"][0]["role"] == "user"
    assert data["memory_trace"][1]["role"] == "assistant"
