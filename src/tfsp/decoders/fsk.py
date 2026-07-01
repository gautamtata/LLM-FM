import numpy as np

from ..encoders.fsk import FREQUENCY_RANGE, MIN_FREQUENCY
from .base import BaseDecoder
from .detection import detect_symbol


class FSKDecoder(BaseDecoder):
    """
    FSK decoder that maps single-tone windows back to characters.

    Mirrors FSKEncoder: candidate frequencies are the 256 points of
    MIN_FREQUENCY + (code / 255) * FREQUENCY_RANGE, and each symbol
    window is matched to the nearest candidate via the correlator bank.
    """

    def __init__(self, tone_duration_ms: int = 100, sample_rate: int = 44100):
        super().__init__(tone_duration_ms, sample_rate)
        codes = np.arange(256)
        self.candidate_freqs = MIN_FREQUENCY + (codes / 255) * FREQUENCY_RANGE

    def decode(self, samples: np.ndarray) -> str:
        """
        Decode an FSK waveform back into text.

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
