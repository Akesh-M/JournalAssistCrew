"""FastAPI application: multi-agent API powered by LangGraph.

Supports single-agent and multi-agent flows with inter-agent conversation:
each agent sees the full message history (user + previous agents) and appends
its response. Uses OpenAI via LangChain.
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from langchain_core.messages import HumanMessage

from backend.graph import get_compiled_graph

# ---------------------------------------------------------------------------
# Request/Response models
# ---------------------------------------------------------------------------

class AgentRequest(BaseModel):
    """Request body for agent execution (single or multi-agent)."""

    agent: str | None = Field(None, description="Single agent id: 'progress' or 'summarize'")
    agents: list[str] | None = Field(
        None,
        description="Ordered list of agents for multi-agent flow (e.g. ['summarize', 'progress'])",
    )
    input: str = Field(..., description="User input for the agent(s)")

    def get_agent_sequence(self) -> list[str]:
        """Resolve agent_sequence: agents if provided, else [agent], else default to progress."""
        if self.agents:
            return [a.strip().lower() for a in self.agents if a.strip()]
        if self.agent:
            return [self.agent.strip().lower()]
        return ["progress"]


class MessageRecord(BaseModel):
    """One message in the conversation (for API response)."""

    role: str  # "user" | "assistant"
    agent: str | None = None  # which agent produced this (if assistant)
    content: str


class AgentResponse(BaseModel):
    """Response from agent execution."""

    agent: str  # last agent that replied (single-agent) or last in chain
    output: str  # last assistant message content (backward compatible)
    messages: list[MessageRecord] | None = Field(
        None,
        description="Full conversation (user + each agent reply) for multi-agent / inter-agent view",
    )


# ---------------------------------------------------------------------------
# App and routes
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Journal Assist Crew API",
    description="Multi-agent API (LangGraph): Progress and Summarize agents with optional inter-agent flow. Powered by OpenAI.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

VALID_AGENTS = {"progress", "summarize"}


@app.get("/agents")
async def list_agents():
    """Return available agent ids and descriptions."""
    return {
        "agents": [
            {"id": "progress", "description": "Analyzes progress and suggests next steps"},
            {"id": "summarize", "description": "Summarizes your text concisely"},
        ],
        "multi_agent": "Use POST /agent/run with body.agents = ['summarize', 'progress'] for inter-agent flow.",
    }


@app.post("/agent/run", response_model=AgentResponse)
async def run_agent(request: AgentRequest):
    """Run one or more agents (LangGraph). Multi-agent: agents run in order; each sees full message history."""
    agent_sequence = request.get_agent_sequence()
    if not agent_sequence:
        raise HTTPException(status_code=400, detail="Provide 'agent' or 'agents'.")
    invalid = [a for a in agent_sequence if a not in VALID_AGENTS]
    if invalid:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown agent(s): {invalid}. Choose from: {list(VALID_AGENTS)}",
        )
    user_input = (request.input or "").strip()
    if not user_input:
        raise HTTPException(status_code=400, detail="'input' must be non-empty.")

    initial_state = {
        "messages": [HumanMessage(content=user_input)],
        "agent_sequence": agent_sequence,
    }
    graph = get_compiled_graph()
    try:
        final_state = await graph.ainvoke(initial_state)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    messages = final_state.get("messages") or []
    last_agent = final_state.get("last_agent") or (agent_sequence[-1] if agent_sequence else "")
    # Last assistant message content (for backward-compatible "output")
    output = ""
    for m in reversed(messages):
        if getattr(m, "type", None) == "ai":
            output = getattr(m, "content", "") or ""
            break
    if not output:
        for m in messages:
            if getattr(m, "name", None):
                output = getattr(m, "content", "") or ""
                break

    # Build message records for inter-agent view
    message_records = []
    for m in messages:
        role = "user" if (getattr(m, "type", None) == "human") else "assistant"
        agent_name = getattr(m, "name", None) if role == "assistant" else None
        content = getattr(m, "content", "") or ""
        message_records.append(MessageRecord(role=role, agent=agent_name, content=content))

    return AgentResponse(
        agent=last_agent,
        output=output,
        messages=message_records,
    )


@app.get("/health")
async def health():
    """Health check."""
    return {"status": "ok", "graph": "langgraph"}
