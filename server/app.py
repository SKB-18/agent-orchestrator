from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
from core.agent import Agent
from core.registry import ToolRegistry
from core.llm import OpenAILikeLLM

app = FastAPI(title="Agent Orchestrator API")

# Setup default registry and dummy tools
registry = ToolRegistry()

def get_current_time(timezone: str) -> str:
    """Returns the current time for a given timezone."""
    import datetime
    return f"The current time in {timezone} is {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

def calculate_sum(a: int, b: int) -> int:
    """Calculates the sum of two integers."""
    return a + b

registry.register("get_current_time", get_current_time)
registry.register("calculate_sum", calculate_sum)

class ChatRequest(BaseModel):
    message: str
    max_steps: int = 5
    
class ChatResponse(BaseModel):
    final_response: str
    memory_trace: List[Dict[str, Any]]

@app.post("/v1/agent/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    # Initialize the LLM client (pointing to a hypothetical local inference server)
    # Since we don't have a real one running on 8001 that speaks chat/completions yet,
    # this will currently hit an LLM Connection Error if called, which is fine for structure.
    llm = OpenAILikeLLM(
        base_url="http://127.0.0.1:8000/v1", 
        tools=[get_current_time, calculate_sum]
    )
    
    agent = Agent(llm=llm, registry=registry)
    
    try:
        final_response = agent.run(user_prompt=request.message, max_steps=request.max_steps)
        return ChatResponse(
            final_response=final_response,
            memory_trace=agent.memory
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
