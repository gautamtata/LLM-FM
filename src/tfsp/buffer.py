"""Token buffering logic."""

from typing import Awaitable, Callable


class TokenBuffer:
    """
    Buffers incoming tokens and triggers encoding when threshold reached.
    """

    def __init__(
        self,
        buffer_size: int,
        on_flush: Callable[[str], Awaitable[None]],
    ):
        """
        Initialize token buffer.

        Args:
            buffer_size: Number of tokens to buffer before flushing
            on_flush: Async callback to invoke when buffer is flushed
        """
        self.buffer_size = buffer_size
        self.on_flush = on_flush
        self._buffer = ""
        self._token_count = 0

    async def add(self, text: str) -> None:
        """
        Add text to buffer, flush if threshold reached.

        Args:
            text: Text chunk to add to buffer
        """
        self._buffer += text
        self._token_count += 1  # Treating each chunk as a token

        if self._token_count >= self.buffer_size:
            await self.flush()

    async def flush(self) -> None:
        """Force flush the buffer."""
        if self._buffer:
            await self.on_flush(self._buffer)
            self._buffer = ""
            self._token_count = 0

