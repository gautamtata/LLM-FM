import numpy as np

from ..encoders.ultrasonic import FREQUENCY_RANGE, MIN_FREQUENCY
from .base import BaseDecoder
from .detection import detect_symbol


class UltrasonicDecoder(BaseDecoder):
    """
    Ultrasonic decoder that maps 15-20kHz tone windows back to characters.

    Mirrors UltrasonicEncoder: 256 candidate frequencies spaced ~19.6 Hz
    apart. Note that at short tone durations the correlation margin
    between neighboring candidates is small (a 5ms window has ~200 Hz of
    intrinsic frequency uncertainty), so decoding is reliable in clean
    loopback but degrades quickly with noise. Longer tones or a smaller
    alphabet trade speed for robustness.
    """

    def __init__(self, tone_duration_ms: int = 5, sample_rate: int = 44100):
        super().__init__(tone_duration_ms, sample_rate)
        codes = np.arange(256)
        self.candidate_freqs = MIN_FREQUENCY + (codes / 255) * FREQUENCY_RANGE

    def decode(self, samples: np.ndarray) -> str:
        """
        Decode an ultrasonic waveform back into text.

        Args:
            samples: Audio samples starting at the first symbol boundary

        Returns:
            The decoded text
        """
        chars = []
        for window in self.segment(samples):
            code = detect_symbol(window, self.candidate_freqs, self.sample_rate)
            chars.append(chr(code))
        return "".join(chars)
