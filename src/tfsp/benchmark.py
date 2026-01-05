"""Benchmark script to compare encoding methods and TTS."""

import asyncio
import os
import sys
import time
from dataclasses import dataclass

from .audio import AudioPlayer, generate_tone
from .encoders import DTMFEncoder, FSKEncoder, UltrasonicEncoder
from .providers import AnthropicProvider, OpenAIProvider
from .tts import ElevenLabsTTS


@dataclass
class BenchmarkResult:
    """Result from a benchmark run."""

    method: str
    text: str
    char_count: int
    encoding_time_ms: float
    playback_time_ms: float
    total_time_ms: float

    @property
    def chars_per_second(self) -> float:
        """Characters processed per second."""
        if self.total_time_ms == 0:
            return 0
        return self.char_count / (self.total_time_ms / 1000)

    @property
    def theoretical_bps(self) -> float | None:
        """Theoretical bits per second (for frequency encoders)."""
        return None


def benchmark_encoder(
    encoder,
    text: str,
    player: AudioPlayer,
    play_audio: bool = False,
) -> BenchmarkResult:
    """
    Benchmark a frequency encoder.

    Args:
        encoder: The encoder to benchmark
        text: Text to encode
        player: Audio player for playback timing
        play_audio: Whether to actually play audio (slower but realistic)

    Returns:
        BenchmarkResult with timing information
    """
    # Time encoding
    start_encode = time.perf_counter()
    frame = encoder.encode(text)
    encode_time = (time.perf_counter() - start_encode) * 1000

    # Calculate playback time (or actually play)
    if play_audio:
        start_play = time.perf_counter()
        player.play_tones(frame.tones)
        playback_time = (time.perf_counter() - start_play) * 1000
    else:
        # Calculate theoretical playback time
        playback_time = sum(tone.duration_ms for tone in frame.tones)

    return BenchmarkResult(
        method=encoder.__class__.__name__,
        text=text,
        char_count=len(text),
        encoding_time_ms=encode_time,
        playback_time_ms=playback_time,
        total_time_ms=encode_time + playback_time,
    )


def benchmark_tts(tts: ElevenLabsTTS, text: str) -> BenchmarkResult:
    """
    Benchmark TTS synthesis.

    Args:
        tts: The TTS engine to benchmark
        text: Text to synthesize

    Returns:
        BenchmarkResult with timing information
    """
    result = tts.synthesize(text)

    # Estimate playback time based on audio duration
    # MP3 at 128kbps, rough estimate
    estimated_playback_ms = len(result.audio_data) / 128 * 8  # Very rough

    return BenchmarkResult(
        method="ElevenLabs TTS",
        text=text,
        char_count=len(text),
        encoding_time_ms=result.synthesis_time_ms,
        playback_time_ms=estimated_playback_ms,
        total_time_ms=result.synthesis_time_ms + estimated_playback_ms,
    )


async def run_benchmark(
    text: str | None = None,
    prompt: str | None = None,
    provider_name: str = "openai",
    include_tts: bool = True,
    play_audio: bool = False,
    verbose: bool = True,
) -> list[BenchmarkResult]:
    """
    Run benchmark comparing all encoding methods.

    Args:
        text: Direct text to benchmark (if provided, skips LLM)
        prompt: Prompt to send to LLM (generates text)
        provider_name: Which LLM provider to use
        include_tts: Whether to include TTS in benchmark
        play_audio: Whether to actually play audio
        verbose: Print progress

    Returns:
        List of BenchmarkResults
    """
    results: list[BenchmarkResult] = []

    # Get text to benchmark
    if text:
        benchmark_text = text
    elif prompt:
        if verbose:
            print(f"Generating text from {provider_name}...")

        if provider_name == "openai":
            api_key = os.environ.get("OPENAI_API_KEY")
            provider = OpenAIProvider(api_key, "gpt-4o-mini")
        else:
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            provider = AnthropicProvider(api_key, "claude-sonnet-4-5-20250929")

        chunks = []
        async for chunk in provider.stream(prompt):
            chunks.append(chunk)
            if verbose:
                print(chunk, end="", flush=True)
        benchmark_text = "".join(chunks)
        if verbose:
            print("\n")
    else:
        benchmark_text = "The quick brown fox jumps over the lazy dog. " * 3

    if verbose:
        print(f"Benchmarking {len(benchmark_text)} characters...")
        print("=" * 60)

    # Initialize components
    player = AudioPlayer()

    # Benchmark each encoder
    encoders = [
        ("DTMF (100ms)", DTMFEncoder(tone_duration_ms=100)),
        ("FSK (100ms)", FSKEncoder(tone_duration_ms=100)),
        ("FSK (50ms)", FSKEncoder(tone_duration_ms=50)),
        ("FSK (10ms)", FSKEncoder(tone_duration_ms=10)),
        ("Ultrasonic (5ms)", UltrasonicEncoder(tone_duration_ms=5)),
        ("Ultrasonic (2ms)", UltrasonicEncoder(tone_duration_ms=2)),
        ("Ultrasonic (1ms)", UltrasonicEncoder(tone_duration_ms=1)),
    ]

    for name, encoder in encoders:
        if verbose:
            print(f"  Testing {name}...", end=" ", flush=True)

        result = benchmark_encoder(encoder, benchmark_text, player, play_audio)
        result.method = name  # Override with descriptive name
        results.append(result)

        if verbose:
            print(f"{result.chars_per_second:.1f} chars/sec")

    # Benchmark TTS if requested
    if include_tts:
        elevenlabs_key = os.environ.get("ELEVENLABS_API_KEY")
        if elevenlabs_key:
            if verbose:
                print(f"  Testing ElevenLabs TTS...", end=" ", flush=True)

            tts = ElevenLabsTTS(api_key=elevenlabs_key)
            result = benchmark_tts(tts, benchmark_text)
            results.append(result)

            if verbose:
                print(f"{result.chars_per_second:.1f} chars/sec")
        elif verbose:
            print("  Skipping TTS (ELEVENLABS_API_KEY not set)")

    return results


def print_benchmark_table(results: list[BenchmarkResult]) -> None:
    """Print benchmark results as a formatted table."""
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80)
    print(
        f"{'Method':<25} {'Encode (ms)':<12} {'Playback (ms)':<14} "
        f"{'Total (ms)':<12} {'Chars/sec':<12}"
    )
    print("-" * 80)

    # Sort by chars/sec descending
    sorted_results = sorted(results, key=lambda r: r.chars_per_second, reverse=True)

    for r in sorted_results:
        print(
            f"{r.method:<25} {r.encoding_time_ms:>10.2f}   {r.playback_time_ms:>12.2f}   "
            f"{r.total_time_ms:>10.2f}   {r.chars_per_second:>10.1f}"
        )

    print("-" * 80)

    # Calculate speedup vs TTS
    tts_result = next((r for r in results if "TTS" in r.method), None)
    if tts_result:
        print("\nSpeedup vs TTS:")
        for r in sorted_results:
            if "TTS" not in r.method:
                speedup = r.chars_per_second / tts_result.chars_per_second
                print(f"  {r.method}: {speedup:.1f}x faster")

    print()


def main() -> None:
    """CLI entry point for benchmark."""
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark TFSP encoding methods")
    parser.add_argument(
        "-t",
        "--text",
        type=str,
        help="Direct text to benchmark",
    )
    parser.add_argument(
        "-p",
        "--prompt",
        type=str,
        help="Prompt to send to LLM to generate text",
    )
    parser.add_argument(
        "--provider",
        choices=["openai", "anthropic"],
        default="openai",
        help="LLM provider to use",
    )
    parser.add_argument(
        "--no-tts",
        action="store_true",
        help="Skip TTS benchmark",
    )
    parser.add_argument(
        "--play",
        action="store_true",
        help="Actually play audio (slower but realistic)",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Minimal output",
    )

    args = parser.parse_args()

    results = asyncio.run(
        run_benchmark(
            text=args.text,
            prompt=args.prompt,
            provider_name=args.provider,
            include_tts=not args.no_tts,
            play_audio=args.play,
            verbose=not args.quiet,
        )
    )

    print_benchmark_table(results)


if __name__ == "__main__":
    main()

