import numpy as np
import sounddevice as sd

from ..encoders.base import ToneEvent
from .tone import generate_tone


class AudioPlayer:
    """Plays audio through system speakers."""

    def __init__(self, sample_rate: int = 44100):
        """
        Initialize audio player.

        Args:
            sample_rate: Audio sample rate (default: 44100)
        """
        self.sample_rate = sample_rate

    def play(self, samples: np.ndarray, blocking: bool = True) -> None:
        """
        Play audio samples.

        Args:
            samples: Audio samples to play (float32, -1 to 1)
            blocking: If True, wait for playback to complete
        """
        sd.play(samples, self.sample_rate)
        if blocking:
            sd.wait()

    def play_tones(self, tones: list[ToneEvent]) -> None:
        """
        Play a sequence of tones.

        Args:
            tones: List of ToneEvent objects to play
        """
        for tone in tones:
            samples = generate_tone(
                tone.frequencies,
                tone.duration_ms,
                self.sample_rate,
            )
            self.play(samples)

