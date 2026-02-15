# Journal Assist Crew

Multi-agent API (FastAPI) powered by **LangGraph**, with a Streamlit UI. **Progress Agent** and **Summarize Agent** support single-agent or **multi-agent flow with inter-agent conversation** (each agent sees prior agents’ outputs). Powered by OpenAI via LangChain.

## Structure

- **Backend** (`backend/`): FastAPI app; **LangGraph** graph in `backend/graph/` (state, nodes, compiled graph). OpenAI via LangChain.
- **Frontend** (`frontend/`): Streamlit — single or multi-agent mode, agent order, full conversation view.
- **Docs** (`docs/DOCUMENTATION.md`): Detailed explanation of architecture, state, inter-agent flow, and API.

## Setup

1. **Clone and enter project**
   ```bash
   cd JournalAssistCrew
   ```

2. **Backend env and install**
   ```bash
   cp env.example .env
   # Edit .env and set OPENAI_API_KEY=sk-...
   python -m venv venv
   source venv/bin/activate   # or: venv\Scripts\activate on Windows
   pip install -r backend/requirements.txt
   ```

3. **Run FastAPI** (from project root)
   ```bash
   uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
   ```

4. **Frontend** (in another terminal, from project root)
   ```bash
   pip install -r frontend/requirements.txt
   streamlit run frontend/app.py
   ```

5. Open the Streamlit URL (e.g. http://localhost:8501). Use **Single agent** or **Multi-agent** (run e.g. Summarize then Progress in order); enter text and click **Run**.

## API

- `GET /agents` — List agents and note on multi-agent.
- `POST /agent/run` — Body: `{"input": "..."}` and either `"agent": "progress"|"summarize"` (single) or `"agents": ["summarize", "progress"]` (multi-agent, order matters). Returns `{"agent", "output", "messages"}` (messages = full conversation for inter-agent view).
- `GET /health` — Health check.

See **docs/DOCUMENTATION.md** for architecture, state, and inter-agent conversation details.

## Env (backend)

| Variable         | Description                    |
|------------------|--------------------------------|
| `OPENAI_API_KEY` | Required. OpenAI API key.      |
| `OPENAI_MODEL`   | Optional. Default: `gpt-4o-mini`. |
