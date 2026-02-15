"""Base agent interface for OpenAI-backed agents."""
from abc import ABC, abstractmethod


class BaseAgent(ABC):
    """Abstract base for all agents using OpenAI."""

    name: str = "base"
    system_prompt: str = ""

    @abstractmethod
    async def run(self, user_input: str) -> str:
        """Process user input and return agent response."""
        ...
