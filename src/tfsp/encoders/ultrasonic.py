"""Ultrasonic encoder for high-speed AI-to-AI communication.

Uses frequencies in the 15-20kHz range which are:
- Inaudible to most adult humans
- Supported by standard audio hardware (44.1kHz sample rate)
- Much faster than audible encoding schemes

With 5ms tones and 256 frequencies, this can achieve ~1.6 KB/sec
which is 100-1000x faster than TTS.
"""

from .base import BaseEncoder, EncodedFrame, ToneEvent

# Ultrasonic frequency range (mostly inaudible to humans)
MIN_FREQUENCY = 15000.0  # 15 kHz - edge of human hearing
MAX_FREQUENCY = 20000.0  # 20 kHz - upper limit of most audio hardware
FREQUENCY_RANGE = MAX_FREQUENCY - MIN_FREQUENCY

# Default tone duration - much shorter than audible encoders
DEFAULT_TONE_DURATION_MS = 5  # 5ms per symbol = 200 symbols/sec


class UltrasonicEncoder(BaseEncoder):
    """
    High-speed ultrasonic encoder for AI-to-AI communication.

    Maps each byte (0-255) to a unique frequency in the 15-20kHz range.
    With 5ms tones, achieves ~1600 bits/sec (200 bytes/sec).

    Inaudible to most humans but perfectly decodable by machines.
    """

    def __init__(self, tone_duration_ms: int = DEFAULT_TONE_DURATION_MS):
        """
        Initialize ultrasonic encoder.

        Args:
            tone_duration_ms: Duration of each tone in milliseconds (default: 5ms)
        """
        self.tone_duration_ms = tone_duration_ms

    def _char_to_frequency(self, char: str) -> float:
        """
        Map a character to its corresponding ultrasonic frequency.

        Uses linear interpolation across 256 values in 15-20kHz range.
        Each frequency is ~19.6 Hz apart, easily distinguishable.
        """
        ascii_code = ord(char)
        ascii_code = max(0, min(255, ascii_code))

        frequency = MIN_FREQUENCY + (ascii_code / 255) * FREQUENCY_RANGE
        return frequency

    def encode(self, text: str) -> EncodedFrame:
        """
        Encode text into ultrasonic tones.

        Each character becomes a single high-frequency tone.
        At 5ms per tone, a 100-char message takes only 500ms.

        Args:
            text: The text to encode

        Returns:
            EncodedFrame containing ultrasonic tone events
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

    @property
    def theoretical_speed_bps(self) -> float:
        """Calculate theoretical bits per second."""
        symbols_per_sec = 1000 / self.tone_duration_ms
        bits_per_symbol = 8  # Full byte per symbol
        return symbols_per_sec * bits_per_symbol

    @property
    def theoretical_speed_description(self) -> str:
        """Human-readable speed description."""
        bps = self.theoretical_speed_bps
        return f"{bps:.0f} bps ({bps/8:.0f} bytes/sec)"

