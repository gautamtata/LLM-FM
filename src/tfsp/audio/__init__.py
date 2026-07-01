"""Audio generation and playback."""

from .tone import generate_tone, render_frame
from .player import AudioPlayer

__all__ = ["generate_tone", "render_frame", "AudioPlayer"]

