from src.tfsp.encoders import DTMFEncoder, FSKEncoder, UltrasonicEncoder


class TestDTMFEncoder:
    """Tests for DTMF encoder."""

    def test_encode_single_char(self):
        """Test encoding a single character."""
        encoder = DTMFEncoder(tone_duration_ms=100)
        frame = encoder.encode("H")

        # 'H' is ASCII 72 = hex 0x48
        # Should produce 2 tones: one for '4', one for '8'
        assert len(frame.tones) == 2
        assert frame.original_text == "H"

    def test_encode_multiple_chars(self):
        """Test encoding multiple characters."""
        encoder = DTMFEncoder(tone_duration_ms=100)
        frame = encoder.encode("Hi")

        # 'H' (0x48) + 'i' (0x69) = 4 tones
        assert len(frame.tones) == 4

    def test_dtmf_frequencies_are_dual_tone(self):
        """Test that DTMF produces dual frequencies."""
        encoder = DTMFEncoder(tone_duration_ms=100)
        frame = encoder.encode("A")

        for tone in frame.tones:
            assert len(tone.frequencies) == 2
            # DTMF frequencies should be in expected ranges
            low_freq = min(tone.frequencies)
            high_freq = max(tone.frequencies)
            assert 697 <= low_freq <= 941
            assert 1209 <= high_freq <= 1633

    def test_tone_duration(self):
        """Test that tone duration is set correctly."""
        encoder = DTMFEncoder(tone_duration_ms=50)
        frame = encoder.encode("X")

        for tone in frame.tones:
            assert tone.duration_ms == 50


class TestFSKEncoder:
    """Tests for FSK encoder."""

    def test_encode_single_char(self):
        """Test encoding a single character."""
        encoder = FSKEncoder(tone_duration_ms=100)
        frame = encoder.encode("A")

        # Should produce 1 tone per character
        assert len(frame.tones) == 1
        assert frame.original_text == "A"

    def test_encode_multiple_chars(self):
        """Test encoding multiple characters."""
        encoder = FSKEncoder(tone_duration_ms=100)
        frame = encoder.encode("Hello")

        # 1 tone per character
        assert len(frame.tones) == 5

    def test_fsk_frequencies_are_single_tone(self):
        """Test that FSK produces single frequencies."""
        encoder = FSKEncoder(tone_duration_ms=100)
        frame = encoder.encode("ABC")

        for tone in frame.tones:
            assert len(tone.frequencies) == 1
            # FSK frequencies should be in 400-2000 Hz range
            assert 400 <= tone.frequencies[0] <= 2000

    def test_frequency_mapping(self):
        """Test that different chars map to different frequencies."""
        encoder = FSKEncoder(tone_duration_ms=100)

        frame_a = encoder.encode("A")  # ASCII 65
        frame_z = encoder.encode("z")  # ASCII 122

        # Higher ASCII code should map to higher frequency
        assert frame_a.tones[0].frequencies[0] < frame_z.tones[0].frequencies[0]

    def test_tone_duration(self):
        """Test that tone duration is set correctly."""
        encoder = FSKEncoder(tone_duration_ms=75)
        frame = encoder.encode("X")

        for tone in frame.tones:
            assert tone.duration_ms == 75


class TestUltrasonicEncoder:
    """Tests for Ultrasonic encoder."""

    def test_encode_single_char(self):
        """Test encoding a single character."""
        encoder = UltrasonicEncoder(tone_duration_ms=5)
        frame = encoder.encode("A")

        assert len(frame.tones) == 1
        assert frame.original_text == "A"

    def test_encode_multiple_chars(self):
        """Test encoding multiple characters."""
        encoder = UltrasonicEncoder(tone_duration_ms=5)
        frame = encoder.encode("Hello")

        assert len(frame.tones) == 5

    def test_ultrasonic_frequencies_in_range(self):
        """Test that ultrasonic produces frequencies in 15-20kHz range."""
        encoder = UltrasonicEncoder(tone_duration_ms=5)
        frame = encoder.encode("ABC")

        for tone in frame.tones:
            assert len(tone.frequencies) == 1
            # Ultrasonic frequencies should be in 15-20kHz range
            assert 15000 <= tone.frequencies[0] <= 20000

    def test_frequency_mapping(self):
        """Test that different chars map to different frequencies."""
        encoder = UltrasonicEncoder(tone_duration_ms=5)

        frame_a = encoder.encode("A")  # ASCII 65
        frame_z = encoder.encode("z")  # ASCII 122

        # Higher ASCII code should map to higher frequency
        assert frame_a.tones[0].frequencies[0] < frame_z.tones[0].frequencies[0]

    def test_fast_tone_duration(self):
        """Test very fast tone durations (key for high speed)."""
        encoder = UltrasonicEncoder(tone_duration_ms=1)
        frame = encoder.encode("X")

        for tone in frame.tones:
            assert tone.duration_ms == 1

    def test_theoretical_speed(self):
        """Test theoretical speed calculation."""
        encoder = UltrasonicEncoder(tone_duration_ms=5)

        # At 5ms per symbol, 8 bits per symbol
        # = 200 symbols/sec * 8 bits = 1600 bps
        assert encoder.theoretical_speed_bps == 1600

    def test_1ms_speed(self):
        """Test speed at 1ms tones."""
        encoder = UltrasonicEncoder(tone_duration_ms=1)

        # At 1ms per symbol = 1000 symbols/sec * 8 bits = 8000 bps
        assert encoder.theoretical_speed_bps == 8000
