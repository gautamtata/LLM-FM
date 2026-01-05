"""OpenAI streaming provider implementation."""

from collections.abc import AsyncGenerator

from openai import AsyncOpenAI

from .base import BaseProvider


class OpenAIProvider(BaseProvider):
    """OpenAI streaming provider using the official SDK."""

    def __init__(self, api_key: str | None, model: str = "gpt-4o-mini"):
        """
        Initialize OpenAI provider.

        Args:
            api_key: OpenAI API key (if None, uses OPENAI_API_KEY env var)
            model: Model to use for completions
        """
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model

    async def stream(self, prompt: str) -> AsyncGenerator[str, None]:
        """
        Stream response tokens from OpenAI.

        Args:
            prompt: The prompt to send to the LLM

        Yields:
            str: Text chunks as they arrive
        """
        stream = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
        )

        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

