"""Progress Agent: analyzes progress and suggests next steps."""
from openai import AsyncOpenAI

from .base import BaseAgent
from backend.config import get_settings


class ProgressAgent(BaseAgent):
    """Agent that evaluates progress and recommends next actions."""

    name = "progress"
    system_prompt = """You are a Progress Agent. Your role is to:
- Analyze the user's current progress (e.g., journal entries, goals, tasks).
- Identify what has been accomplished and what is pending.
- Suggest clear, actionable next steps to maintain or accelerate progress.
- Be encouraging and specific. Respond in a structured, readable way."""

    def __init__(self):
        settings = get_settings()
        self._client = AsyncOpenAI(api_key=settings.openai_api_key)
        self._model = settings.openai_model

    async def run(self, user_input: str) -> str:
        if not user_input.strip():
            return "Please provide some context about your current progress (e.g., journal notes, goals, or tasks) so I can analyze and suggest next steps."

        response = await self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_input},
            ],
        )
        return response.choices[0].message.content or ""
