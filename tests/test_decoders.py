"""Loopback tests: encode -> render to samples -> decode -> compare.

These run entirely in numpy (no audio device). "Clean" tests assert
perfect roundtrips; "noisy" tests add white Gaussian noise at a given
SNR and document how each scheme degrades — in particular that short
ultrasonic tones only survive clean channels, while DTMF shrugs off
heavy noise.
"""

import numpy as np

from src.tfsp.audio.tone import render_frame
from src.tfsp.decoders import DTMFDecoder, FSKDecoder, UltrasonicDecoder
from src.tfsp.encoders import DTMFEncoder, FSKEncoder, UltrasonicEncoder

SAMPLE_RATE = 44100
MESSAGE = "The quick brown fox jumps over the lazy dog"


def add_noise(samples: np.ndarray, snr_db: float, seed: int = 42) -> np.ndarray:
    """Add white Gaussian noise at the given signal-to-noise ratio."""
    rng = np.random.default_rng(seed)
    signal_power = np.mean(samples.astype(np.float64) ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = rng.normal(0, np.sqrt(noise_power), len(samples))
    return samples + noise


def char_accuracy(expected: str, actual: str) -> float:
    """Fraction of characters decoded correctly (position-wise)."""
    if not expected:
        return 1.0
    matches = sum(e == a for e, a in zip(expected, actual))
    return matches / len(expected)


def roundtrip(encoder, decoder, text: str, snr_db: float | None = None) -> str:
    """Encode text, render to audio samples, optionally add noise, decode."""
    frame = encoder.encode(text)
    samples = render_frame(frame, SAMPLE_RATE)
    if snr_db is not None:
        samples = add_noise(samples, snr_db)
    return decoder.decode(samples)


class TestFSKLoopback:
    """FSK: 256 tones across 400-2000 Hz (~6.3 Hz spacing)."""

    def test_clean_roundtrip_100ms(self):
        encoder = FSKEncoder(tone_duration_ms=100)
        decoder = FSKDecoder(tone_duration_ms=100)
        assert roundtrip(encoder, decoder, MESSAGE) == MESSAGE

    def test_clean_roundtrip_50ms(self):
        encoder = FSKEncoder(tone_duration_ms=50)
        decoder = FSKDecoder(tone_duration_ms=50)
        assert roundtrip(encoder, decoder, MESSAGE) == MESSAGE

    def test_clean_roundtrip_10ms(self):
        encoder = FSKEncoder(tone_duration_ms=10)
        decoder = FSKDecoder(tone_duration_ms=10)
        assert roundtrip(encoder, decoder, MESSAGE) == MESSAGE

    def test_all_256_codes(self):
        """Every possible byte value must roundtrip."""
        text = "".join(chr(i) for i in range(256))
        encoder = FSKEncoder(tone_duration_ms=50)
        decoder = FSKDecoder(tone_duration_ms=50)
        assert roundtrip(encoder, decoder, text) == text

    def test_noisy_roundtrip_100ms(self):
        """Long FSK tones survive moderate noise."""
        encoder = FSKEncoder(tone_duration_ms=100)
        decoder = FSKDecoder(tone_duration_ms=100)
        decoded = roundtrip(encoder, decoder, MESSAGE, snr_db=10)
        assert char_accuracy(MESSAGE, decoded) >= 0.95


class TestUltrasonicLoopback:
    """Ultrasonic: 256 tones across 15-20 kHz (~19.6 Hz spacing)."""

    def test_clean_roundtrip_5ms(self):
        encoder = UltrasonicEncoder(tone_duration_ms=5)
        decoder = UltrasonicDecoder(tone_duration_ms=5)
        assert roundtrip(encoder, decoder, MESSAGE) == MESSAGE

    def test_clean_roundtrip_2ms(self):
        encoder = UltrasonicEncoder(tone_duration_ms=2)
        decoder = UltrasonicDecoder(tone_duration_ms=2)
        assert roundtrip(encoder, decoder, MESSAGE) == MESSAGE

    def test_clean_roundtrip_1ms(self):
        encoder = UltrasonicEncoder(tone_duration_ms=1)
        decoder = UltrasonicDecoder(tone_duration_ms=1)
        assert roundtrip(encoder, decoder, MESSAGE) == MESSAGE

    def test_all_256_codes(self):
        """Every possible byte value must roundtrip."""
        text = "".join(chr(i) for i in range(256))
        encoder = UltrasonicEncoder(tone_duration_ms=5)
        decoder = UltrasonicDecoder(tone_duration_ms=5)
        assert roundtrip(encoder, decoder, text) == text

    def test_noisy_roundtrip_needs_high_snr(self):
        """5ms tones at 19.6 Hz spacing survive mild noise (30 dB SNR)."""
        encoder = UltrasonicEncoder(tone_duration_ms=5)
        decoder = UltrasonicDecoder(tone_duration_ms=5)
        decoded = roundtrip(encoder, decoder, MESSAGE, snr_db=30)
        assert char_accuracy(MESSAGE, decoded) >= 0.95

    def test_noise_degrades_short_tones(self):
        """Document the physics: 1ms tones break down before 5ms tones do.

        At the same SNR, shorter windows integrate less signal energy, so
        the correlation margin between 19.6 Hz-spaced candidates shrinks
        and errors appear sooner.
        """
        snr_db = 10
        acc_5ms = char_accuracy(
            MESSAGE,
            roundtrip(
                UltrasonicEncoder(tone_duration_ms=5),
                UltrasonicDecoder(tone_duration_ms=5),
                MESSAGE,
                snr_db=snr_db,
            ),
        )
        acc_1ms = char_accuracy(
            MESSAGE,
            roundtrip(
                UltrasonicEncoder(tone_duration_ms=1),
                UltrasonicDecoder(tone_duration_ms=1),
                MESSAGE,
                snr_db=snr_db,
            ),
        )
        assert acc_5ms >= acc_1ms
        # 1ms tones at 10 dB SNR should be visibly degraded
        assert acc_1ms < 1.0


class TestDTMFLoopback:
    """DTMF: 16 symbols on the standard telephone frequency grid."""

    def test_clean_roundtrip_100ms(self):
        encoder = DTMFEncoder(tone_duration_ms=100)
        decoder = DTMFDecoder(tone_duration_ms=100)
        assert roundtrip(encoder, decoder, MESSAGE) == MESSAGE

    def test_clean_roundtrip_50ms(self):
        encoder = DTMFEncoder(tone_duration_ms=50)
        decoder = DTMFDecoder(tone_duration_ms=50)
        assert roundtrip(encoder, decoder, MESSAGE) == MESSAGE

    def test_all_256_codes(self):
        """Every possible byte value must roundtrip through hex pairs."""
        text = "".join(chr(i) for i in range(256))
        encoder = DTMFEncoder(tone_duration_ms=50)
        decoder = DTMFDecoder(tone_duration_ms=50)
        assert roundtrip(encoder, decoder, text) == text

    def test_noisy_roundtrip_0db(self):
        """DTMF's tiny alphabet makes it extremely noise-tolerant."""
        encoder = DTMFEncoder(tone_duration_ms=100)
        decoder = DTMFDecoder(tone_duration_ms=100)
        decoded = roundtrip(encoder, decoder, MESSAGE, snr_db=0)
        assert char_accuracy(MESSAGE, decoded) >= 0.95

    def test_trailing_partial_symbol_dropped(self):
        """An odd trailing hex digit must not crash the decoder."""
        encoder = DTMFEncoder(tone_duration_ms=50)
        decoder = DTMFDecoder(tone_duration_ms=50)
        frame = encoder.encode("Hi")
        samples = render_frame(frame, SAMPLE_RATE)
        # Chop off the last symbol window, leaving 3 hex digits
        truncated = samples[: -decoder.samples_per_symbol]
        assert decoder.decode(truncated) == "H"


class TestDecoderEdgeCases:
    """Shared decoder behavior."""

    def test_empty_input(self):
        decoder = FSKDecoder(tone_duration_ms=100)
        assert decoder.decode(np.zeros(0, dtype=np.float32)) == ""

    def test_input_shorter_than_one_symbol(self):
        decoder = FSKDecoder(tone_duration_ms=100)
        assert decoder.decode(np.zeros(10, dtype=np.float32)) == ""

    def test_render_frame_empty(self):
        encoder = FSKEncoder(tone_duration_ms=100)
        frame = encoder.encode("")
        assert len(render_frame(frame, SAMPLE_RATE)) == 0

    def test_render_frame_length(self):
        """Rendered length must be exactly symbols * samples_per_symbol."""
        encoder = UltrasonicEncoder(tone_duration_ms=5)
        frame = encoder.encode("Hello")
        samples = render_frame(frame, SAMPLE_RATE)
        expected = 5 * int(SAMPLE_RATE * 5 / 1000)
        assert len(samples) == expected

    def test_short_tones_get_fade_envelope(self):
        """Regression: tones under 20ms previously got no fade at all,
        leaving a rectangular window (clicks + spectral splatter)."""
        from src.tfsp.audio.tone import generate_tone

        samples = generate_tone([17500.0], 5, SAMPLE_RATE)
        # First and last samples should be faded toward zero
        assert abs(samples[0]) < 0.1
        assert abs(samples[-1]) < 0.1
        # But the middle should carry full-scale signal
        assert np.max(np.abs(samples)) > 0.9
