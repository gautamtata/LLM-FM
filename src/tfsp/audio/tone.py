import numpy as np

from ..encoders.base import EncodedFrame


def generate_tone(
    frequencies: list[float],
    duration_ms: int,
    sample_rate: int = 44100,
) -> np.ndarray:
    """
    Generate audio samples for one or more simultaneous frequencies.

    Args:
        frequencies: List of frequencies in Hz
        duration_ms: Duration in milliseconds
        sample_rate: Audio sample rate (default: 44100)

    Returns:
        numpy array of audio samples (float32, -1 to 1)
    """
    num_samples = int(sample_rate * duration_ms / 1000)
    # np.arange/sample_rate, NOT np.linspace(0, T, N): linspace includes the
    # endpoint, giving a sample step of T/(N-1) instead of 1/fs. That makes
    # every tone sharp by a factor of N/(N-1) — negligible for long tones but
    # several symbol-widths of error for 5ms ultrasonic tones.
    t = (np.arange(num_samples) / sample_rate).astype(np.float32)

    # Sum all frequencies (for DTMF dual-tone)
    signal = np.zeros_like(t)
    for freq in frequencies:
        signal += np.sin(2 * np.pi * freq * t)

    # Normalize by number of frequencies to prevent clipping
    if len(frequencies) > 0:
        signal = signal / len(frequencies)

    # Apply envelope to avoid clicks: fade in/out over 10ms, or 20% of the
    # tone for short tones. A fixed 10ms fade would be skipped entirely for
    # tones under 20ms (i.e. all ultrasonic tones), leaving a rectangular
    # window whose spectral sidelobes splatter into the audible band.
    fade_samples = min(int(sample_rate * 0.01), int(num_samples * 0.2))
    if fade_samples > 0:
        fade_in = np.linspace(0, 1, fade_samples, dtype=np.float32)
        fade_out = np.linspace(1, 0, fade_samples, dtype=np.float32)
        signal[:fade_samples] *= fade_in
        signal[-fade_samples:] *= fade_out

    return signal.astype(np.float32)


def render_frame(frame: EncodedFrame, sample_rate: int = 44100) -> np.ndarray:
    """
    Render an entire encoded frame to one continuous waveform.

    Concatenates the samples for every tone in the frame. Because each
    tone occupies exactly int(sample_rate * duration_ms / 1000) samples,
    a decoder that knows the tone duration can segment the waveform back
    into per-symbol windows without any synchronization signal.

    Args:
        frame: The encoded frame to render
        sample_rate: Audio sample rate (default: 44100)

    Returns:
        numpy array of audio samples (float32, -1 to 1)
    """
    if not frame.tones:
        return np.zeros(0, dtype=np.float32)

    return np.concatenate(
        [
            generate_tone(tone.frequencies, tone.duration_ms, sample_rate)
            for tone in frame.tones
        ]
    )

