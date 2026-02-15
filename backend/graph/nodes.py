"""Graph nodes: progress agent, summarize agent, and orchestrator.

Each agent node receives the current message history (including prior agents'
outputs) and appends its response, enabling inter-agent conversation.
"""
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI

from backend.config import get_settings
from backend.graph.state import AgentState

# ---------------------------------------------------------------------------
# Agent system prompts (same behavior as before, now used inside the graph)
# ---------------------------------------------------------------------------

PROGRESS_SYSTEM = """You are a Progress Agent. Your role is to:
- Analyze the user's current progress (e.g., journal entries, goals, tasks).
- Identify what has been accomplished and what is pending.
- Suggest clear, actionable next steps to maintain or accelerate progress.
- Be encouraging and specific. Respond in a structured, readable way."""

SUMMARIZE_SYSTEM = """You are a Summarize Agent. Your role is to:
- Summarize the user's input clearly and concisely.
- Preserve key points, decisions, and outcomes.
- Use bullet points or short paragraphs when helpful.
- Keep the summary focused and easy to scan."""

# ---------------------------------------------------------------------------
# LLM (shared, created once per process)
# ---------------------------------------------------------------------------


def _get_llm():
    settings = get_settings()
    return ChatOpenAI(
        model=settings.openai_model,
        api_key=settings.openai_api_key,
        temperature=0.3,
    )


# ---------------------------------------------------------------------------
# Agent nodes: each takes state, invokes LLM with agent-specific system
# prompt and full message history, returns state update (new message).
# ---------------------------------------------------------------------------


def _invoke_agent(messages: list, system_prompt: str, agent_name: str) -> AIMessage:
    llm = _get_llm()
    # Build prompt with system + conversation so far
    from langchain_core.messages import SystemMessage

    full_messages = [SystemMessage(content=system_prompt)] + list(messages)
    response = llm.invoke(full_messages)
    content = response.content if hasattr(response, "content") else str(response)
    return AIMessage(content=content or "", name=agent_name)


def progress_node(state: AgentState) -> dict:
    """Progress Agent node: analyze progress and suggest next steps."""
    messages = state.get("messages") or []
    if not messages:
        return {
            "messages": [
                AIMessage(
                    content="Please provide some context about your current progress (e.g., journal notes, goals, or tasks).",
                    name="progress",
                )
            ],
            "last_agent": "progress",
        }
    reply = _invoke_agent(messages, PROGRESS_SYSTEM, "progress")
    return {"messages": [reply], "last_agent": "progress"}


def summarize_node(state: AgentState) -> dict:
    """Summarize Agent node: produce a concise summary of the conversation so far."""
    messages = state.get("messages") or []
    if not messages:
        return {
            "messages": [
                AIMessage(
                    content="Please provide the text or notes you want summarized.",
                    name="summarize",
                )
            ],
            "last_agent": "summarize",
        }
    reply = _invoke_agent(messages, SUMMARIZE_SYSTEM, "summarize")
    return {"messages": [reply], "last_agent": "summarize"}


# ---------------------------------------------------------------------------
# Router: runs the next agent in agent_sequence and returns updated state.
# Conditional edge: if more agents remain, loop to run_next_agent else END.
# ---------------------------------------------------------------------------

AGENT_NODES = {
    "progress": progress_node,
    "summarize": summarize_node,
}


def run_next_agent(state: AgentState) -> dict:
    """Run the next agent in agent_sequence and return state update."""
    agent_sequence = list(state.get("agent_sequence") or [])
    if not agent_sequence:
        return {"last_agent": "", "agent_sequence": []}
    next_agent_id = agent_sequence[0].strip().lower()
    rest = agent_sequence[1:]
    if next_agent_id not in AGENT_NODES:
        return {"agent_sequence": rest, "last_agent": next_agent_id}
    node_fn = AGENT_NODES[next_agent_id]
    update = node_fn(state)
    update["agent_sequence"] = rest
    return update


def should_continue(state: AgentState) -> str:
    """Return next node: 'run_next_agent' if more agents, else '__end__'."""
    rest = state.get("agent_sequence") or []
    return "run_next_agent" if rest else "__end__"
