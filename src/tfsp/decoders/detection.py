"""Tone detection via a correlator bank.

Rather than locating an FFT peak and interpolating (which is limited by
the time-bandwidth product: a T-second window smears energy over ~1/T Hz),
we exploit that the transmitter's alphabet is known. Each symbol window is
correlated against a complex exponential at every candidate frequency and
the strongest response wins.

For a single tone in additive white noise this is the maximum-likelihood
detector, and it is what lets short windows (e.g. 5ms ultrasonic tones)
be decoded even though the candidate spacing (~19.6 Hz) is far below the
window's FFT resolution (~200 Hz). The margin between neighboring
candidates shrinks as tones get shorter, so noise tolerance degrades —
see the decoder benchmarks in tests/test_decoders.py.
"""

import numpy as np


def correlate_bank(
    window: np.ndarray,
    candidate_freqs: np.ndarray,
    sample_rate: int,
) -> np.ndarray:
    """
    Correlate a sample window against a bank of candidate frequencies.

    Computes |sum(x[n] * exp(-2j*pi*f*n/fs))| for each candidate f —
    the DTFT of the window evaluated exactly at the candidate frequencies.

    Args:
        window: Audio samples for one symbol
        candidate_freqs: 1-D array of candidate frequencies in Hz
        sample_rate: Audio sample rate

    Returns:
        1-D array of correlation magnitudes, one per candidate
    """
    t = np.arange(len(window)) / sample_rate
    # (num_candidates, num_samples) matrix of complex exponentials
    exponentials = np.exp(-2j * np.pi * np.outer(candidate_freqs, t))
    return np.abs(exponentials @ window.astype(np.float64))


def detect_symbol(
    window: np.ndarray,
    candidate_freqs: np.ndarray,
    sample_rate: int,
) -> int:
    """
    Detect which candidate frequency best matches a symbol window.

    Args:
        window: Audio samples for one symbol
        candidate_freqs: 1-D array of candidate frequencies in Hz
        sample_rate: Audio sample rate

    Returns:
        Index into candidate_freqs of the strongest correlation
    """
    magnitudes = correlate_bank(window, candidate_freqs, sample_rate)
    return int(np.argmax(magnitudes))
