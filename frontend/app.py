"""Streamlit UI for Journal Assist Crew: single or multi-agent flow (LangGraph)."""
import streamlit as st
import requests

DEFAULT_API_BASE = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Journal Assist Crew",
    page_icon="📔",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    .stTextArea textarea { font-size: 1rem; }
    .agent-badge { display: inline-block; padding: 0.2em 0.5em; border-radius: 4px; font-size: 0.85em; margin-right: 0.5em; }
    .msg-user { background: #e3f2fd; padding: 0.5rem 0.75rem; border-radius: 8px; margin: 0.25rem 0; }
    .msg-agent { background: #f5f5f5; padding: 0.5rem 0.75rem; border-radius: 8px; margin: 0.25rem 0; border-left: 3px solid #1e88e5; }
</style>
""", unsafe_allow_html=True)

st.title("📔 Journal Assist Crew")
st.caption("Multi-agent flow (LangGraph): Progress & Summarize — inter-agent conversation supported")

# Mode: single agent vs multi-agent
mode = st.radio(
    "Mode",
    options=["single", "multi"],
    format_func=lambda x: "Single agent" if x == "single" else "Multi-agent (run agents in order)",
    horizontal=True,
)

agents_option = [
    ("progress", "Progress Agent", "Analyze progress and next steps"),
    ("summarize", "Summarize Agent", "Summarize text concisely"),
]

if mode == "single":
    selected = st.radio(
        "Select agent",
        options=[a[0] for a in agents_option],
        format_func=lambda x: next((f"{n} — {d}" for i, n, d in agents_option if i == x), x),
        horizontal=True,
    )
    agent_sequence = [selected]
else:
    st.markdown("**Agents to run (in order)** — each agent sees the previous outputs.")
    selected_multi = st.multiselect(
        "Order matters: e.g. Summarize then Progress",
        options=[a[0] for a in agents_option],
        default=["summarize", "progress"],
        format_func=lambda x: next((n for i, n, _ in agents_option if i == x), x),
    )
    agent_sequence = selected_multi

# Input
user_input = st.text_area(
    "Input",
    placeholder="Paste your journal notes, goals, or text here…",
    height=180,
    label_visibility="collapsed",
)

run = st.button("Run", type="primary")

api_base = st.session_state.get("api_base", DEFAULT_API_BASE)

if run:
    if not (user_input and user_input.strip()):
        st.warning("Please enter some text.")
    elif mode == "multi" and not agent_sequence:
        st.warning("Select at least one agent for multi-agent flow.")
    else:
        with st.spinner("Running agent(s)…"):
            try:
                payload = {"input": user_input.strip()}
                if mode == "single":
                    payload["agent"] = agent_sequence[0]
                else:
                    payload["agents"] = agent_sequence
                r = requests.post(
                    f"{api_base}/agent/run",
                    json=payload,
                    timeout=120,
                )
                r.raise_for_status()
                data = r.json()
                st.success("Done")

                # Show full conversation when messages are returned (inter-agent)
                messages = data.get("messages") or []
                if messages:
                    st.subheader("Conversation")
                    for msg in messages:
                        role = msg.get("role", "")
                        agent = msg.get("agent")
                        content = msg.get("content", "")
                        if role == "user":
                            st.markdown(f'<div class="msg-user">👤 **You**</div><div class="msg-user">{content}</div>', unsafe_allow_html=True)
                        else:
                            label = f"🤖 **{agent or 'Assistant'}**" if agent else "🤖 **Assistant**"
                            st.markdown(f'<div class="msg-agent">{label}</div><div class="msg-agent">{content}</div>', unsafe_allow_html=True)
                else:
                    st.subheader("Output")
                    st.markdown(data.get("output", ""))
            except requests.exceptions.ConnectionError:
                st.error(f"Cannot reach API at {api_base}. Start the FastAPI server first.")
            except requests.exceptions.HTTPError as e:
                st.error(f"API error: {e.response.status_code} — {e.response.text}")
            except Exception as e:
                st.error(str(e))

with st.sidebar:
    st.subheader("Settings")
    api_url = st.text_input("API base URL", value=st.session_state.get("api_base", DEFAULT_API_BASE), key="api_url")
    if api_url:
        st.session_state["api_base"] = api_url.rstrip("/")
    if st.button("Check API"):
        try:
            r = requests.get(f"{st.session_state.get('api_base', DEFAULT_API_BASE)}/health", timeout=5)
            if r.ok:
                j = r.json()
                st.success(f"API up — {j.get('graph', '') or 'ok'}")
            else:
                st.warning(f"API returned {r.status_code}")
        except Exception as e:
            st.error(f"Cannot reach API: {e}")
