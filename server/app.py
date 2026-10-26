from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import math

from core.agent import Agent
from core.registry import ToolRegistry
from core.llm import OpenAILikeLLM
from core.handoff import HandoffCoordinator, Specialist
from core.reflection import ReflectionAgent
from core.debate import DebateCoordinator

app = FastAPI(
    title="Agent Orchestrator API",
    description="Production Multi-Agent Orchestration Framework with ReAct, Handoff, Reflection, and Debate.",
    version="2.0.0"
)

# -------------------------------------------------------------
# Tool Registries & Domain Functions
# -------------------------------------------------------------
math_registry = ToolRegistry()
search_registry = ToolRegistry()
code_registry = ToolRegistry()
general_registry = ToolRegistry()

def add_numbers(a: float, b: float) -> float:
    """Adds two numbers together."""
    return a + b

def multiply_numbers(a: float, b: float) -> float:
    """Multiplies two numbers."""
    return a * b

def compute_power(base: float, exponent: float) -> float:
    """Computes base raised to the power of exponent."""
    return math.pow(base, exponent)

def search_knowledge(query: str) -> str:
    """Searches the knowledge base for factual information."""
    kb = {
        "kv cache": "Key-Value caching stores attention tensors across autoregressive decoding steps to reduce computation.",
        "speculative decoding": "Speculative decoding uses a draft model to generate candidates verified in parallel by a target model.",
        "continuous batching": "Continuous batching or iteration-level scheduling packs active requests dynamically on every decode step."
    }
    for k, v in kb.items():
        if k in query.lower():
            return v
    return f"Knowledge lookup for '{query}': High-performance distributed AI and systems engineering pattern."

def python_eval(expression: str) -> str:
    """Safely evaluates a basic python mathematical expression."""
    allowed = {"__builtins__": None, "math": math, "abs": abs, "min": min, "max": max, "round": round}
    try:
        return str(eval(expression, allowed, {}))
    except Exception as e:
        return f"Execution error: {str(e)}"

math_registry.register("add_numbers", add_numbers)
math_registry.register("multiply_numbers", multiply_numbers)
math_registry.register("compute_power", compute_power)

search_registry.register("search_knowledge", search_knowledge)
code_registry.register("python_eval", python_eval)

general_registry.register("add_numbers", add_numbers)
general_registry.register("search_knowledge", search_knowledge)

# Default LLM instances (targeting local Inference Lab on port 8002)
default_llm = OpenAILikeLLM(base_url="http://127.0.0.1:8002/v1")

# Initialize Multi-Agent Handoff
handoff_coordinator = HandoffCoordinator(router_llm=default_llm)
handoff_coordinator.register_specialist(Specialist("math", "Solves arithmetic and equations", default_llm, math_registry))
handoff_coordinator.register_specialist(Specialist("search", "Retrieves factual knowledge", default_llm, search_registry))
handoff_coordinator.register_specialist(Specialist("code", "Executes and verifies python code", default_llm, code_registry))

# Initialize Reflection & Debate Engines
reflection_agent = ReflectionAgent(generator_llm=default_llm)
debate_coordinator = DebateCoordinator(proposer_llm=default_llm, skeptic_llm=default_llm)


# -------------------------------------------------------------
# Request & Response Schemas
# -------------------------------------------------------------
class ChatRequest(BaseModel):
    message: str
    max_steps: int = Field(default=5, ge=1, le=20)

class ChatResponse(BaseModel):
    final_response: str
    memory_trace: List[Dict[str, Any]]

class HandoffRequest(BaseModel):
    query: str
    max_steps: int = Field(default=5, ge=1, le=20)

class ReflectionRequest(BaseModel):
    problem: str

class DebateRequest(BaseModel):
    topic: str
    rounds: int = Field(default=2, ge=1, le=5)


# -------------------------------------------------------------
# Endpoints
# -------------------------------------------------------------
@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "agent-orchestrator",
        "specialists": list(handoff_coordinator.specialists.keys()),
        "backend": "http://127.0.0.1:8002/v1"
    }

@app.post("/v1/agent/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    agent = Agent(llm=default_llm, registry=general_registry)
    try:
        final_response = agent.run(user_prompt=request.message, max_steps=request.max_steps)
        return ChatResponse(
            final_response=final_response,
            memory_trace=agent.memory
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/agent/handoff")
async def handoff_endpoint(request: HandoffRequest):
    try:
        result = handoff_coordinator.orchestrate(user_query=request.query, max_steps=request.max_steps)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/agent/reflect")
async def reflect_endpoint(request: ReflectionRequest):
    try:
        result = reflection_agent.solve(problem=request.problem)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/agent/debate")
async def debate_endpoint(request: DebateRequest):
    try:
        result = debate_coordinator.conduct_debate(topic=request.topic, rounds=request.rounds)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
