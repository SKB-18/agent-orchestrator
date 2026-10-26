from typing import Dict, Any, List, Optional
from core.llm import BaseLLM


class DebateCoordinator:
    """
    Multi-Agent Debate Framework.
    
    Orchestrates dialectical debate between divergent agent personas:
    - Proposer: Defends thesis / proposed approach.
    - Skeptic: Attacks assumptions, presents trade-offs and counter-evidence.
    - Judge: Evaluates arguments over N rounds and renders a balanced final verdict.
    """
    def __init__(
        self, 
        proposer_llm: BaseLLM, 
        skeptic_llm: BaseLLM, 
        judge_llm: Optional[BaseLLM] = None
    ):
        self.proposer_llm = proposer_llm
        self.skeptic_llm = skeptic_llm
        self.judge_llm = judge_llm or proposer_llm

    def conduct_debate(self, topic: str, rounds: int = 2) -> Dict[str, Any]:
        transcript: List[Dict[str, Any]] = []

        proposer_history = ""
        skeptic_history = ""

        for r in range(1, rounds + 1):
            # Proposer Turn
            proposer_prompt = (
                f"Topic: {topic}\n"
                f"You are the Proposer advocating strongly for the proposition.\n"
                f"Current debate history:\n{skeptic_history}\n"
                f"Round {r}: Present your strongest arguments, address counterpoints, and defend your position."
            )
            proposer_resp = self.proposer_llm.generate(prompt=proposer_prompt, memory=[])
            proposer_arg = proposer_resp.get("content", "").strip()
            
            transcript.append({
                "round": r,
                "role": "Proposer",
                "argument": proposer_arg
            })
            proposer_history += f"\nRound {r} Proposer: {proposer_arg}\n"

            # Skeptic Turn
            skeptic_prompt = (
                f"Topic: {topic}\n"
                f"You are the Skeptic / Devil's Advocate scrutinizing the proposition.\n"
                f"Current debate history:\n{proposer_history}\n"
                f"Round {r}: Challenge assumptions, highlight risks, trade-offs, and flaws in the proposer's argument."
            )
            skeptic_resp = self.skeptic_llm.generate(prompt=skeptic_prompt, memory=[])
            skeptic_arg = skeptic_resp.get("content", "").strip()

            transcript.append({
                "round": r,
                "role": "Skeptic",
                "argument": skeptic_arg
            })
            skeptic_history += f"\nRound {r} Skeptic: {skeptic_arg}\n"

        # Judge Verdict
        judge_prompt = (
            f"Topic: {topic}\n"
            f"You are the impartial Presiding Judge.\n"
            f"Full Debate Transcript:\n"
            + "\n".join(f"[{item['role']} - Round {item['round']}]: {item['argument']}" for item in transcript)
            + "\n\nSynthesize the debate and deliver your Final Ruling:\n"
            f"1. Key strengths of Proposer\n"
            f"2. Key strengths of Skeptic\n"
            f"3. Final Balanced Consensus Verdict"
        )
        judge_resp = self.judge_llm.generate(prompt=judge_prompt, memory=[])
        final_verdict = judge_resp.get("content", "").strip()

        return {
            "topic": topic,
            "rounds_conducted": rounds,
            "transcript": transcript,
            "final_verdict": final_verdict
        }
