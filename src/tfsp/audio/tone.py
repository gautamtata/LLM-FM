import numpy as np


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
    t = np.linspace(0, duration_ms / 1000, num_samples, dtype=np.float32)

    # Sum all frequencies (for DTMF dual-tone)
    signal = np.zeros_like(t)
    for freq in frequencies:
        signal += np.sin(2 * np.pi * freq * t)

    # Normalize by number of frequencies to prevent clipping
    if len(frequencies) > 0:
        signal = signal / len(frequencies)

    # Apply envelope to avoid clicks (10ms fade in/out)
    fade_samples = int(sample_rate * 0.01)
    if fade_samples > 0 and len(signal) > 2 * fade_samples:
        fade_in = np.linspace(0, 1, fade_samples, dtype=np.float32)
        fade_out = np.linspace(1, 0, fade_samples, dtype=np.float32)
        signal[:fade_samples] *= fade_in
        signal[-fade_samples:] *= fade_out

    return signal.astype(np.float32)

