from typing import Dict, Any, List, Optional
from core.agent import Agent
from core.registry import ToolRegistry
from core.llm import BaseLLM


class Specialist:
    """Represents a domain specialist agent with dedicated tools and system instructions."""
    def __init__(self, name: str, description: str, llm: BaseLLM, registry: ToolRegistry):
        self.name = name
        self.description = description
        self.agent = Agent(llm=llm, registry=registry)

    def execute(self, task: str, max_steps: int = 5) -> Dict[str, Any]:
        response = self.agent.run(user_prompt=task, max_steps=max_steps)
        return {
            "specialist": self.name,
            "response": response,
            "memory_trace": list(self.agent.memory)
        }


class HandoffCoordinator:
    """
    Multi-Agent Handoff Orchestrator.
    Routes user tasks to domain-specialist agents and aggregates execution traces.
    """
    def __init__(self, router_llm: BaseLLM):
        self.router_llm = router_llm
        self.specialists: Dict[str, Specialist] = {}

    def register_specialist(self, specialist: Specialist):
        self.specialists[specialist.name.lower()] = specialist

    def route_query(self, query: str) -> str:
        """
        Determines the most appropriate specialist for the given query.
        Uses keyword-heuristics combined with router LLM decision.
        """
        query_lower = query.lower()
        
        # Fast intent classification
        math_keywords = ["calculate", "math", "sum", "multiply", "divide", "add", "subtract", "formula", "fibonacci", "factorial", "compute"]
        code_keywords = ["python", "code", "function", "script", "syntax", "bug", "execute", "eval", "algorithm"]
        search_keywords = ["who is", "what is", "where is", "when did", "search", "find", "history", "lookup", "tell me about"]
        
        if any(kw in query_lower for kw in math_keywords) and "math" in self.specialists:
            return "math"
        if any(kw in query_lower for kw in code_keywords) and "code" in self.specialists:
            return "code"
        if any(kw in query_lower for kw in search_keywords) and "search" in self.specialists:
            return "search"

        # Ask router LLM
        prompt = (
            f"Given the user request: '{query}'\n"
            f"Available specialists: {list(self.specialists.keys())}\n"
            f"Respond with ONLY the name of the best specialist."
        )
        resp = self.router_llm.generate(prompt=prompt, memory=[])
        selected = resp.get("content", "").strip().lower()
        
        for name in self.specialists:
            if name in selected:
                return name
                
        # Default fallback to first specialist
        return next(iter(self.specialists.keys())) if self.specialists else "general"

    def orchestrate(self, user_query: str, max_steps: int = 5) -> Dict[str, Any]:
        target_name = self.route_query(user_query)
        specialist = self.specialists.get(target_name)
        
        if not specialist:
            return {
                "status": "error",
                "target_specialist": target_name,
                "error": f"Specialist '{target_name}' is not registered."
            }

        result = specialist.execute(user_query, max_steps=max_steps)
        return {
            "status": "success",
            "routed_specialist": target_name,
            "final_answer": result["response"],
            "execution_trace": result["memory_trace"]
        }
