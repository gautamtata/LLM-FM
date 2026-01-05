from .base import BaseEncoder, EncodedFrame, ToneEvent

# DTMF frequency mapping
# Standard DTMF keypad layout:
#        1209 Hz   1336 Hz   1477 Hz   1633 Hz
# 697 Hz    1         2         3         A
# 770 Hz    4         5         6         B
# 852 Hz    7         8         9         C
# 941 Hz    *         0         #         D

LOW_FREQUENCIES = {
    "1": 697,
    "2": 697,
    "3": 697,
    "A": 697,
    "4": 770,
    "5": 770,
    "6": 770,
    "B": 770,
    "7": 852,
    "8": 852,
    "9": 852,
    "C": 852,
    "*": 941,
    "0": 941,
    "#": 941,
    "D": 941,
}

HIGH_FREQUENCIES = {
    "1": 1209,
    "4": 1209,
    "7": 1209,
    "*": 1209,
    "2": 1336,
    "5": 1336,
    "8": 1336,
    "0": 1336,
    "3": 1477,
    "6": 1477,
    "9": 1477,
    "#": 1477,
    "A": 1633,
    "B": 1633,
    "C": 1633,
    "D": 1633,
}

# Map hex digits to DTMF symbols
# 0-9 map directly, A-F map to A-D, *, #
HEX_TO_DTMF = {
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
    "a": "A",
    "b": "B",
    "c": "C",
    "d": "D",
    "e": "*",
    "f": "#",
    "A": "A",
    "B": "B",
    "C": "C",
    "D": "D",
    "E": "*",
    "F": "#",
}


class DTMFEncoder(BaseEncoder):
    """
    DTMF encoder that converts text to dual-tone frequencies.

    Each character is converted to its hex representation (2 digits),
    and each hex digit maps to a DTMF symbol with corresponding
    low and high frequency pair.
    """

    def __init__(self, tone_duration_ms: int = 100):
        """
        Initialize DTMF encoder.

        Args:
            tone_duration_ms: Duration of each tone in milliseconds
        """
        self.tone_duration_ms = tone_duration_ms

    def _char_to_dtmf_symbols(self, char: str) -> list[str]:
        """Convert a character to its DTMF symbol representation."""
        # Get ASCII code and convert to 2-digit hex
        ascii_code = ord(char)
        hex_str = f"{ascii_code:02x}"

        # Map each hex digit to DTMF symbol
        return [HEX_TO_DTMF[digit] for digit in hex_str]

    def _symbol_to_frequencies(self, symbol: str) -> list[float]:
        """Get the frequency pair for a DTMF symbol."""
        return [
            float(LOW_FREQUENCIES[symbol]),
            float(HIGH_FREQUENCIES[symbol]),
        ]

    def encode(self, text: str) -> EncodedFrame:
        """
        Encode text into DTMF tones.

        Each character becomes 2 DTMF tones (one per hex digit).

        Args:
            text: The text to encode

        Returns:
            EncodedFrame containing dual-tone events
        """
        tones: list[ToneEvent] = []

        for char in text:
            symbols = self._char_to_dtmf_symbols(char)
            for symbol in symbols:
                frequencies = self._symbol_to_frequencies(symbol)
                tones.append(
                    ToneEvent(
                        frequencies=frequencies,
                        duration_ms=self.tone_duration_ms,
                    )
                )

        return EncodedFrame(tones=tones, original_text=text)

