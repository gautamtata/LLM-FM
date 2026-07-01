from abc import ABC, abstractmethod

import numpy as np


class BaseDecoder(ABC):
    """Base class for frequency decoders."""

    def __init__(self, tone_duration_ms: int, sample_rate: int = 44100):
        """
        Initialize decoder.

        Args:
            tone_duration_ms: Duration of each tone in milliseconds
                (must match the encoder's setting)
            sample_rate: Audio sample rate (default: 44100)
        """
        self.tone_duration_ms = tone_duration_ms
        self.sample_rate = sample_rate

    @property
    def samples_per_symbol(self) -> int:
        """Number of audio samples per symbol window."""
        # Must match generate_tone() exactly so windows align
        return int(self.sample_rate * self.tone_duration_ms / 1000)

    def segment(self, samples: np.ndarray) -> list[np.ndarray]:
        """
        Split a waveform into per-symbol windows.

        Assumes the waveform starts at the first symbol boundary and
        symbols are back-to-back (as produced by render_frame). A
        trailing partial window is dropped.

        Args:
            samples: Audio samples to segment

        Returns:
            List of per-symbol sample windows
        """
        n = self.samples_per_symbol
        num_symbols = len(samples) // n
        return [samples[i * n : (i + 1) * n] for i in range(num_symbols)]

    @abstractmethod
    def decode(self, samples: np.ndarray) -> str:
        """
        Decode a waveform back into text.

        Args:
            samples: Audio samples (float32/float64, as produced by
                render_frame or captured from a microphone)

        Returns:
            The decoded text
        """
        pass
