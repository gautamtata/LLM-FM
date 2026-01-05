import numpy as np

from src.tfsp.audio import generate_tone


class TestToneGeneration:
    """Tests for tone generation."""

    def test_generate_single_frequency(self):
        """Test generating a single frequency tone."""
        samples = generate_tone([440.0], duration_ms=100, sample_rate=44100)

        # Should return float32 array
        assert samples.dtype == np.float32

        # Should have correct number of samples
        expected_samples = int(44100 * 0.1)  # 100ms at 44100 Hz
        assert len(samples) == expected_samples

    def test_generate_dual_frequency(self):
        """Test generating a dual frequency tone (DTMF style)."""
        samples = generate_tone([697.0, 1209.0], duration_ms=100, sample_rate=44100)

        assert samples.dtype == np.float32
        assert len(samples) == int(44100 * 0.1)

    def test_samples_normalized(self):
        """Test that samples are in -1 to 1 range."""
        samples = generate_tone([440.0, 880.0], duration_ms=100, sample_rate=44100)

        assert np.all(samples >= -1.0)
        assert np.all(samples <= 1.0)

    def test_envelope_applied(self):
        """Test that fade envelope is applied (no clicks)."""
        samples = generate_tone([440.0], duration_ms=100, sample_rate=44100)

        # First and last samples should be near zero due to fade
        assert abs(samples[0]) < 0.01
        assert abs(samples[-1]) < 0.01

    def test_different_sample_rate(self):
        """Test with different sample rate."""
        samples = generate_tone([440.0], duration_ms=100, sample_rate=22050)

        expected_samples = int(22050 * 0.1)
        assert len(samples) == expected_samples

    def test_empty_frequencies(self):
        """Test with empty frequency list."""
        samples = generate_tone([], duration_ms=100, sample_rate=44100)

        # Should return zeros
        assert np.all(samples == 0)

