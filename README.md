# Agent Orchestrator

This is Repo 2 of the AI Project Portfolio Roadmap.

It implements a foundational ReAct-style Agent Orchestrator capable of looping through thoughts, extracting JSON tool calls, and maintaining conversational memory.

## Architecture
- **`core/agent.py`**: The main execution loop. It tracks memory and controls step limits.
- **`core/registry.py`**: A dynamic tool registry. Maps strings to Python callables.
- **`core/parser.py`**: Robust JSON extractor designed to pull tool calls from messy LLM text and markdown code blocks.
- **`core/prompt.py`**: Dynamic schema generator. Inspects python tools and builds system prompts.
- **`core/llm.py`**: An `OpenAILikeLLM` client connecting this agent to `/v1/chat/completions` API endpoints (like `inference-lab`).
- **`server/app.py`**: A FastAPI server exposing the agent as a network service.

## Testing
Comprehensive TDD tests cover parsing, tool extraction, and loop execution.
```bash
python -m pytest tests/
```
