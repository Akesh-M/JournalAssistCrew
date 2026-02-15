"""Build and compile the LangGraph multi-agent graph.

Flow: START -> run_next_agent -> (if more agents) run_next_agent else END.
Each run_next_agent call runs one agent from agent_sequence; that agent sees
the full message history (user + previous agents) and appends its reply.
"""
from langgraph.graph import StateGraph, START, END

from backend.graph.state import AgentState
from backend.graph.nodes import run_next_agent, should_continue

# Optional: persist state across turns (e.g. for multi-turn chat later).
# memory = MemorySaver()
# compiled = builder.compile(checkpointer=memory)


def build_graph():
    """Build the multi-agent StateGraph and return the compiled graph."""
    builder = StateGraph(AgentState)

    builder.add_node("run_next_agent", run_next_agent)
    builder.add_edge(START, "run_next_agent")
    builder.add_conditional_edges("run_next_agent", should_continue, {"run_next_agent": "run_next_agent", "__end__": END})

    return builder.compile()


# Single compiled instance for the API (no checkpointer for now).
_compiled = None


def get_compiled_graph():
    """Return the compiled graph, building it once."""
    global _compiled
    if _compiled is None:
        _compiled = build_graph()
    return _compiled
