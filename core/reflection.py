from typing import Dict, Any, List, Optional
from core.llm import BaseLLM


class ReflectionAgent:
    """
    Self-Reflection and Critique Engine (GSM8K-style iterative reasoning).
    
    Implements a 3-phase reasoning loop:
    1. Generator: Drafts initial solution and step-by-step chain of thought.
    2. Critic: Evaluates draft for arithmetic mistakes, missing constraints, or logical flaws.
    3. Refiner: Synthesizes final corrected answer conditioned on critic feedback.
    """
    def __init__(self, generator_llm: BaseLLM, critic_llm: Optional[BaseLLM] = None):
        self.generator_llm = generator_llm
        self.critic_llm = critic_llm or generator_llm

    def solve(self, problem: str) -> Dict[str, Any]:
        trace: List[Dict[str, str]] = []

        # 1. Draft Phase
        draft_prompt = (
            f"Solve the following reasoning problem carefully with step-by-step logic:\n"
            f"Problem: {problem}\n"
            f"Show your work and conclude with the final answer."
        )
        draft_resp = self.generator_llm.generate(prompt=draft_prompt, memory=[])
        draft_content = draft_resp.get("content", "").strip()
        trace.append({"phase": "draft", "content": draft_content})

        # 2. Critique Phase
        critique_prompt = (
            f"You are an expert verification auditor. Analyze the draft solution for errors:\n"
            f"Problem: {problem}\n"
            f"Draft Solution: {draft_content}\n\n"
            f"Check each step for mathematical accuracy, logical consistency, and edge cases.\n"
            f"List any flaws or confirm correctness."
        )
        critique_resp = self.critic_llm.generate(prompt=critique_prompt, memory=[])
        critique_content = critique_resp.get("content", "").strip()
        trace.append({"phase": "critique", "content": critique_content})

        # 3. Refinement Phase
        refine_prompt = (
            f"You are finalizing the solution based on peer review:\n"
            f"Problem: {problem}\n"
            f"Initial Draft: {draft_content}\n"
            f"Auditor Critique: {critique_content}\n\n"
            f"Provide the definitive, corrected step-by-step derivation and final boxed answer."
        )
        refine_resp = self.generator_llm.generate(prompt=refine_prompt, memory=[])
        refined_content = refine_resp.get("content", "").strip()
        trace.append({"phase": "refinement", "content": refined_content})

        return {
            "problem": problem,
            "draft": draft_content,
            "critique": critique_content,
            "final_solution": refined_content,
            "trace": trace
        }
