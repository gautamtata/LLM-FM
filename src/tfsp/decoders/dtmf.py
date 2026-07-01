import numpy as np

from ..encoders.dtmf import HIGH_FREQUENCIES, LOW_FREQUENCIES
from .base import BaseDecoder
from .detection import detect_symbol

# The four row and four column frequencies of the DTMF grid
ROW_FREQUENCIES = [697.0, 770.0, 852.0, 941.0]
COLUMN_FREQUENCIES = [1209.0, 1336.0, 1477.0, 1633.0]

# (low, high) frequency pair -> DTMF symbol
_PAIR_TO_SYMBOL = {
    (float(LOW_FREQUENCIES[sym]), float(HIGH_FREQUENCIES[sym])): sym
    for sym in LOW_FREQUENCIES
}

# DTMF symbol -> hex digit (inverse of the encoder's HEX_TO_DTMF)
_SYMBOL_TO_HEX = {
    "0": "0",
    "1": "1",
    "2": "2",
    "3": "3",
    "4": "4",
    "5": "5",
    "6": "6",
    "7": "7",
    "8": "8",
    "9": "9",
    "A": "a",
    "B": "b",
    "C": "c",
    "D": "d",
    "*": "e",
    "#": "f",
}


class DTMFDecoder(BaseDecoder):
    """
    DTMF decoder that maps dual-tone windows back to characters.

    Mirrors DTMFEncoder: each symbol window contains one row frequency
    and one column frequency, detected independently (a 2x4-way choice
    each, which is why DTMF is so robust). Each detected symbol yields
    one hex digit; consecutive digit pairs are reassembled into
    characters.
    """

    def __init__(self, tone_duration_ms: int = 100, sample_rate: int = 44100):
        super().__init__(tone_duration_ms, sample_rate)
        self.row_freqs = np.array(ROW_FREQUENCIES)
        self.column_freqs = np.array(COLUMN_FREQUENCIES)

    def _detect_dtmf_symbol(self, window: np.ndarray) -> str:
        """Detect the DTMF symbol in a single window."""
        row_idx = detect_symbol(window, self.row_freqs, self.sample_rate)
        col_idx = detect_symbol(window, self.column_freqs, self.sample_rate)
        pair = (ROW_FREQUENCIES[row_idx], COLUMN_FREQUENCIES[col_idx])
        return _PAIR_TO_SYMBOL[pair]

    def decode(self, samples: np.ndarray) -> str:
        """
        Decode a DTMF waveform back into text.

        Args:
            samples: Audio samples starting at the first symbol boundary

        Returns:
            The decoded text (a trailing unpaired hex digit is dropped)
        """
        hex_digits = [
            _SYMBOL_TO_HEX[self._detect_dtmf_symbol(window)]
            for window in self.segment(samples)
        ]

        chars = []
        for i in range(0, len(hex_digits) - 1, 2):
            code = int(hex_digits[i] + hex_digits[i + 1], 16)
            chars.append(chr(code))
        return "".join(chars)
