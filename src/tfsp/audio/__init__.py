"""Audio generation and playback."""

from .tone import generate_tone
from .player import AudioPlayer

__all__ = ["generate_tone", "AudioPlayer"]

