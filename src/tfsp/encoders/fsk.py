from .base import BaseEncoder, EncodedFrame, ToneEvent

# FSK frequency range
MIN_FREQUENCY = 400.0  # Hz
MAX_FREQUENCY = 2000.0  # Hz
FREQUENCY_RANGE = MAX_FREQUENCY - MIN_FREQUENCY


class FSKEncoder(BaseEncoder):
    """
    FSK encoder that maps characters to single frequencies.

    Each character's ASCII code (0-255) is linearly interpolated
    to a frequency in the range 400-2000 Hz.
    """

    def __init__(self, tone_duration_ms: int = 100):
        """
        Initialize FSK encoder.

        Args:
            tone_duration_ms: Duration of each tone in milliseconds
        """
        self.tone_duration_ms = tone_duration_ms

    def _char_to_frequency(self, char: str) -> float:
        """
        Map a character to its corresponding frequency.

        Uses linear interpolation:
        frequency = MIN_FREQ + (ascii_code / 255) * RANGE
        """
        ascii_code = ord(char)
        # Clamp to 0-255 range for safety
        ascii_code = max(0, min(255, ascii_code))

        frequency = MIN_FREQUENCY + (ascii_code / 255) * FREQUENCY_RANGE
        return frequency

    def encode(self, text: str) -> EncodedFrame:
        """
        Encode text into FSK tones.

        Each character becomes a single-frequency tone.

        Args:
            text: The text to encode

        Returns:
            EncodedFrame containing single-tone events
        """
        tones: list[ToneEvent] = []

        for char in text:
            frequency = self._char_to_frequency(char)
            tones.append(
                ToneEvent(
                    frequencies=[frequency],
                    duration_ms=self.tone_duration_ms,
                )
            )

        return EncodedFrame(tones=tones, original_text=text)

