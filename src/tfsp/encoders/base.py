from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class ToneEvent:
    """A single tone or chord to play."""

    frequencies: list[float]  # Hz (1 for FSK, 2 for DTMF)
    duration_ms: int  # How long to play


@dataclass
class EncodedFrame:
    """A sequence of tones representing encoded data."""

    tones: list[ToneEvent]
    original_text: str  # For debugging/display


class BaseEncoder(ABC):
    """Base class for frequency encoders."""

    @abstractmethod
    def encode(self, text: str) -> EncodedFrame:
        """
        Encode text into a sequence of tones.

        Args:
            text: The text to encode

        Returns:
            EncodedFrame containing the tones to play
        """
        pass

