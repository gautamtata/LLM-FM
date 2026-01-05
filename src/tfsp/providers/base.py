"""Base class for LLM providers."""

from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator


class BaseProvider(ABC):
    """Base class for LLM providers."""

    @abstractmethod
    async def stream(self, prompt: str) -> AsyncGenerator[str, None]:
        """
        Stream response tokens from the LLM.

        Args:
            prompt: The prompt to send to the LLM

        Yields:
            str: Text chunks as they arrive (may be partial tokens)
        """
        # Abstract async generator - yield for type checker
        yield ""
        raise NotImplementedError

