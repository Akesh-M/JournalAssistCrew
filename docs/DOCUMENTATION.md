# Journal Assist Crew — Detailed Documentation

This document explains the **multi-agent flow** implementation using **LangGraph**, how **inter-agent conversation** works, and what each part of the codebase does.

---

## 1. What Was Done

### 1.1 High-Level Summary

- **Backend** was refactored from a simple “one agent per request” design to a **LangGraph-based multi-agent graph**. The API still supports a single selected agent, but now also supports **multiple agents in sequence** with shared conversation state.
- **Inter-agent conversation**: When you run more than one agent (e.g. Summarize then Progress), each agent sees **all previous messages** (the user’s input plus every prior agent’s reply). So the second agent can build on the first agent’s output.
- **Frontend** (Streamlit) was updated to allow:
  - **Single-agent mode**: choose one agent and get one reply.
  - **Multi-agent mode**: choose an **ordered list** of agents; the backend runs them in that order and returns the full conversation (user + each agent’s reply).
- **OpenAI** is used via **LangChain** (`langchain-openai`, `ChatOpenAI`) for both Progress and Summarize agents. Configuration (API key, model) is unchanged (e.g. `.env` with `OPENAI_API_KEY`, `OPENAI_MODEL`).

### 1.2 Why LangGraph?

- **Stateful graph**: The flow is a **directed graph** with a **shared state** (messages + control data). That fits multi-step and multi-agent workflows.
- **Clear flow**: Entry → run one agent → optionally run the next agent → … → end. Adding more agents or branches later is straightforward.
- **Inter-agent context**: State holds the full **message list**; each node appends to it, so later agents automatically see earlier replies.
- **Extensible**: You can add routing, human-in-the-loop, or tools by adding nodes and edges without rewriting the whole pipeline.

---

## 2. Architecture

### 2.1 Component Overview

| Component        | Role                                                                 |
|-----------------|----------------------------------------------------------------------|
| **FastAPI**     | HTTP API: receives requests, builds initial state, invokes the graph, returns response. |
| **LangGraph**   | Builds and runs the multi-agent graph (state schema, nodes, edges).  |
| **State**       | Holds `messages`, `agent_sequence`, `last_agent` (see below).        |
| **Nodes**       | `run_next_agent` (orchestrator), plus logic for Progress and Summarize. |
| **Streamlit**   | UI: single vs multi-agent, agent order, input, and conversation display. |
| **OpenAI**      | Used via LangChain’s `ChatOpenAI` inside the agent nodes.             |

### 2.2 Directory Layout (Relevant Parts)

```
JournalAssistCrew/
├── backend/
│   ├── main.py              # FastAPI app, /agent/run, /agents, /health
│   ├── config.py            # Settings (OpenAI key, model) from .env
│   ├── graph/
│   │   ├── __init__.py
│   │   ├── state.py         # AgentState (messages, agent_sequence, etc.)
│   │   ├── nodes.py         # progress_node, summarize_node, run_next_agent, should_continue
│   │   └── graph.py         # build_graph(), get_compiled_graph()
│   └── requirements.txt    # fastapi, uvicorn, langgraph, langchain-*, pydantic
├── frontend/
│   ├── app.py               # Streamlit: single/multi agent, input, conversation
│   └── requirements.txt
├── docs/
│   └── DOCUMENTATION.md     # This file
├── env.example
└── README.md
```

The old `backend/agents/` (ProgressAgent, SummarizeAgent classes) is **no longer used** by the API; their behavior is implemented inside `backend/graph/nodes.py` as graph nodes.

---

## 3. State and Inter-Agent Conversation

### 3.1 State Schema (`backend/graph/state.py`)

The graph state is a **TypedDict** with:

- **`messages`**  
  - Type: `Annotated[list[BaseMessage], add_messages]`.  
  - Holds the full conversation: user message(s) and each agent’s reply.  
  - The `add_messages` reducer **appends** new messages instead of replacing the list, so every node sees the full history.

- **`agent_sequence`**  
  - Type: `list[str]`.  
  - Ordered list of agent ids to run, e.g. `["summarize", "progress"]`.  
  - The orchestrator node runs the first agent, then the next, and so on until the list is empty.

- **`last_agent`**  
  - Which agent last produced a reply (used in the API response).

- **`final_output`**  
  - Optional; reserved for a single “final” string if needed later.

So: **inter-agent conversation** is implemented by (1) keeping all messages in one list and (2) running agents in sequence, each reading that list and appending its own reply.

### 3.2 How the Graph Runs

1. **API** receives `input` and either `agent` (single) or `agents` (multi).
2. It builds **initial state**:
   - `messages = [HumanMessage(content=user_input)]`
   - `agent_sequence = [agent]` or `agents` (e.g. `["summarize", "progress"]`)
3. The compiled graph is **invoked** with this state.
4. **Flow**:
   - **START** → **run_next_agent**.
   - **run_next_agent**:
     - Pops the first id from `agent_sequence`.
     - Calls the corresponding agent node (Progress or Summarize) with the **current state** (so it sees all messages so far).
     - The agent node calls the LLM with its system prompt + current messages, then **appends** one `AIMessage` to state and sets `last_agent`.
     - Puts the rest of `agent_sequence` back into state.
   - **Conditional edge**: `should_continue(state)`:
     - If `agent_sequence` is not empty → go back to **run_next_agent** (next agent runs with updated messages).
     - If empty → **END**.
5. **API** reads the final state: collects `messages` and the last assistant message content, and returns them (see “API contract” below).

So the **second agent** (e.g. Progress) literally sees the **first agent’s** (e.g. Summarize) reply in `messages`, enabling true inter-agent conversation.

---

## 4. Graph Definition (Code-Level)

### 4.1 Building the Graph (`backend/graph/graph.py`)

- **StateGraph(AgentState)** is created.
- **Single node**: `run_next_agent` (orchestrator).
- **Edges**:
  - **START** → **run_next_agent**.
  - **run_next_agent** → **conditional**:
    - If more agents in sequence → **run_next_agent** (loop).
    - Else → **END**.
- The graph is **compiled** once and reused via `get_compiled_graph()`.

### 4.2 Nodes (`backend/graph/nodes.py`)

- **progress_node(state)**  
  - Reads `state["messages"]`, calls `ChatOpenAI` with the Progress system prompt and those messages.  
  - Returns `{"messages": [AIMessage(...)], "last_agent": "progress"}`.

- **summarize_node(state)**  
  - Same pattern for the Summarize agent.

- **run_next_agent(state)**  
  - Pops the next agent id from `agent_sequence`, calls the corresponding node function, merges the returned state update and the remaining `agent_sequence`.

- **should_continue(state)**  
  - Returns `"run_next_agent"` if `agent_sequence` is non-empty, else `"__end__"` (map to END in the graph).

LLM creation uses `backend/config.py` (e.g. `get_settings()`) so **OpenAI API key and model** are configured there and via `.env`.

---

## 5. API Contract

### 5.1 Endpoints

- **GET /agents**  
  - Returns list of agents (id, description) and a note that multi-agent is supported via `agents`.

- **POST /agent/run**  
  - **Body**:
    - `input` (required): string.
    - `agent` (optional): single agent id (`"progress"` or `"summarize"`).
    - `agents` (optional): list of agent ids in order, e.g. `["summarize", "progress"]`.
  - If `agents` is provided, it is used as `agent_sequence`; otherwise `[agent]` is used. So you can do single-agent or multi-agent from the same endpoint.

- **GET /health**  
  - Returns `{"status": "ok", "graph": "langgraph"}` (or similar).

### 5.2 Response Shape

- **agent**: string — id of the last agent that replied.
- **output**: string — content of the **last** assistant message (backward compatible with the old “single output” behavior).
- **messages**: list of `{ "role": "user" | "assistant", "agent": string | null, "content": string }` — full conversation (user + each agent reply), so the client can show inter-agent conversation.

---

## 6. Frontend (Streamlit)

- **Mode**:
  - **Single agent**: one radio selection; request sends `agent` + `input`.
  - **Multi-agent**: multiselect for “agents to run (in order)”; request sends `agents` + `input`.
- **Input**: one text area; same for both modes.
- **Response**:
  - If the API returns `messages`, the app shows the full **conversation** (user + each agent with a label).
  - Otherwise it falls back to showing `output` only.

So the UI directly reflects the **inter-agent conversation** when multiple agents are run.

---

## 7. Configuration and Running

- **Environment**: Copy `env.example` to `.env` and set `OPENAI_API_KEY` (and optionally `OPENAI_MODEL`). The backend loads this via `config.py` / `pydantic-settings`.
- **Backend**: From project root, `uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000`.
- **Frontend**: `streamlit run frontend/app.py`.

Dependencies are in `backend/requirements.txt` (FastAPI, LangGraph, LangChain, etc.) and `frontend/requirements.txt` (Streamlit, requests).

---

## 8. Extending the System

- **New agents**: In `nodes.py`, add a new node (e.g. `reflection_node`) and register it in `AGENT_NODES`. Add the id to `VALID_AGENTS` in `main.py` and to the Streamlit options.
- **Routing**: You can add a dedicated “router” node that decides the next agent from the conversation (e.g. based on last message or a classifier) and sets `agent_sequence` accordingly.
- **Human-in-the-loop**: LangGraph supports interrupts; you can add a node that pauses for user input and then continues the graph.
- **Persistence**: Use a LangGraph checkpointer (e.g. `MemorySaver` or a DB-backed one) in `build_graph()` so state (and thus conversation) can span multiple HTTP requests or sessions.

This document describes the current behavior and design; for step-by-step setup and quick reference, see the project **README**.
