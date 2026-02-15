"""State schema for the multi-agent LangGraph.

Messages are stored so that inter-agent conversation is possible: each agent
sees the full history (user input + previous agents' outputs) and appends its
response. agent_sequence drives which agent(s) run and in what order.
"""
from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict, total=False):
    """State shared across all nodes in the graph."""

    # Conversation history; new messages are appended via add_messages reducer.
    messages: Annotated[list[BaseMessage], add_messages]
    # Ordered list of agent ids to run (e.g. ["summarize", "progress"]).
    agent_sequence: list[str]
    # Last agent that produced output (for API response).
    last_agent: str
    # Optional final concatenated output (for backward compatibility).
    final_output: str
