"""Anthropic streaming provider implementation."""

from collections.abc import AsyncGenerator

from anthropic import AsyncAnthropic

from .base import BaseProvider


class AnthropicProvider(BaseProvider):
    """Anthropic streaming provider using the official SDK."""

    def __init__(self, api_key: str | None, model: str = "claude-sonnet-4-5-20250929"):
        """
        Initialize Anthropic provider.

        Args:
            api_key: Anthropic API key (if None, uses ANTHROPIC_API_KEY env var)
            model: Model to use for completions
        """
        self.client = AsyncAnthropic(api_key=api_key)
        self.model = model

    async def stream(self, prompt: str) -> AsyncGenerator[str, None]:
        """
        Stream response tokens from Anthropic.

        Args:
            prompt: The prompt to send to the LLM

        Yields:
            str: Text chunks as they arrive
        """
        stream = await self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
        )

        async for event in stream:
            if event.type == "content_block_delta" and hasattr(event.delta, "text"):
                yield event.delta.text

