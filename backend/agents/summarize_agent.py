"""Summarize Agent: produces concise summaries of provided content."""
from openai import AsyncOpenAI

from .base import BaseAgent
from backend.config import get_settings


class SummarizeAgent(BaseAgent):
    """Agent that summarizes text clearly and concisely."""

    name = "summarize"
    system_prompt = """You are a Summarize Agent. Your role is to:
- Summarize the user's input clearly and concisely.
- Preserve key points, decisions, and outcomes.
- Use bullet points or short paragraphs when helpful.
- Keep the summary focused and easy to scan."""

    def __init__(self):
        settings = get_settings()
        self._client = AsyncOpenAI(api_key=settings.openai_api_key)
        self._model = settings.openai_model

    async def run(self, user_input: str) -> str:
        if not user_input.strip():
            return "Please provide the text or notes you want summarized."

        response = await self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_input},
            ],
        )
        return response.choices[0].message.content or ""
