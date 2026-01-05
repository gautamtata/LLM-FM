"""CLI entry point for Token-to-Frequency Streaming Protocol."""

import argparse
import asyncio
import os
import sys

from dotenv import load_dotenv

from src.tfsp.audio import AudioPlayer
from src.tfsp.buffer import TokenBuffer
from src.tfsp.encoders import DTMFEncoder, FSKEncoder, UltrasonicEncoder
from src.tfsp.providers import AnthropicProvider, OpenAIProvider

load_dotenv()


async def run_stream(args: argparse.Namespace) -> None:
    """Run the TFSP streaming pipeline."""
    # Initialize provider
    if args.openai:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print("Error: OPENAI_API_KEY environment variable not set", file=sys.stderr)
            sys.exit(1)
        model = args.model or "gpt-4o-mini"
        provider = OpenAIProvider(api_key, model)
    else:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            print(
                "Error: ANTHROPIC_API_KEY environment variable not set",
                file=sys.stderr,
            )
            sys.exit(1)
        model = args.model or "claude-sonnet-4-5-20250929"
        provider = AnthropicProvider(api_key, model)

    # Initialize encoder
    if args.dtmf:
        encoder = DTMFEncoder(tone_duration_ms=args.tone_duration)
        encoding_name = "DTMF"
    elif args.fsk:
        encoder = FSKEncoder(tone_duration_ms=args.tone_duration)
        encoding_name = "FSK"
    elif args.ultrasonic:
        # Default to faster tones for ultrasonic
        duration = args.tone_duration if args.tone_duration != 100 else 5
        encoder = UltrasonicEncoder(tone_duration_ms=duration)
        encoding_name = "Ultrasonic"
    else:  # TTS mode
        await run_tts(args, provider)
        return

    # Initialize audio
    player = AudioPlayer()

    # Callback for when buffer flushes
    async def on_buffer_flush(text: str) -> None:
        if args.verbose:
            print(f"\n[ENCODE] '{text}'")

        frame = encoder.encode(text)

        if args.verbose:
            for tone in frame.tones:
                freqs = ", ".join(f"{f:.0f}Hz" for f in tone.frequencies)
                print(f"  → [{freqs}] {tone.duration_ms}ms")

        player.play_tones(frame.tones)

    # Initialize buffer
    buffer = TokenBuffer(args.buffer_tokens, on_buffer_flush)

    # Stream and process
    print(f"Streaming from {provider.__class__.__name__} using {encoding_name}...")
    print(f"Tone duration: {encoder.tone_duration_ms}ms")
    print("-" * 40)

    async for chunk in provider.stream(args.prompt):
        print(chunk, end="", flush=True)
        await buffer.add(chunk)

    # Flush any remaining
    await buffer.flush()

    print()
    print("-" * 40)
    print("Done.")


async def run_tts(args: argparse.Namespace, provider) -> None:
    """Run TTS mode - stream from LLM then synthesize speech."""
    from src.tfsp.tts import ElevenLabsTTS

    api_key = os.environ.get("ELEVENLABS_API_KEY")
    if not api_key:
        print("Error: ELEVENLABS_API_KEY environment variable not set", file=sys.stderr)
        sys.exit(1)

    tts = ElevenLabsTTS(api_key=api_key)

    print(f"Streaming from {provider.__class__.__name__}...")
    print("-" * 40)

    # Collect full response
    chunks = []
    async for chunk in provider.stream(args.prompt):
        print(chunk, end="", flush=True)
        chunks.append(chunk)

    full_text = "".join(chunks)
    print()
    print("-" * 40)

    print("Synthesizing speech with ElevenLabs...")
    result = tts.synthesize_and_play(full_text)
    print(f"TTS synthesis took {result.synthesis_time_ms:.0f}ms")
    print("Done.")


def run_benchmark(args: argparse.Namespace) -> None:
    """Run benchmark comparing all methods."""
    from src.tfsp.benchmark import main as benchmark_main

    # Re-parse with benchmark args
    sys.argv = ["benchmark"]
    if args.prompt:
        sys.argv.extend(["-p", args.prompt])
    if args.benchmark_text:
        sys.argv.extend(["-t", args.benchmark_text])
    if args.anthropic:
        sys.argv.extend(["--provider", "anthropic"])
    if args.no_tts:
        sys.argv.append("--no-tts")
    if args.play_audio:
        sys.argv.append("--play")

    benchmark_main()


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Token-to-Frequency Streaming Protocol",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Frequency encoding modes
  uv run main.py --dtmf --openai -p "Count from 1 to 10"
  uv run main.py --fsk --anthropic -p "What is the capital of France?"
  uv run main.py --ultrasonic --openai -p "Hello" --tone-duration 2

  # TTS mode (for comparison)
  uv run main.py --tts --openai -p "Hello world"

  # Benchmark all methods
  uv run main.py --benchmark -p "Tell me a short joke"
  uv run main.py --benchmark -t "The quick brown fox jumps over the lazy dog"
        """,
    )

    # Prompt
    parser.add_argument(
        "-p",
        "--prompt",
        type=str,
        help="The prompt to send to the LLM",
    )

    # Provider (mutually exclusive, required for streaming)
    provider_group = parser.add_mutually_exclusive_group()
    provider_group.add_argument(
        "-o",
        "--openai",
        action="store_true",
        help="Use OpenAI API (requires OPENAI_API_KEY env var)",
    )
    provider_group.add_argument(
        "-a",
        "--anthropic",
        action="store_true",
        help="Use Anthropic API (requires ANTHROPIC_API_KEY env var)",
    )

    # Encoder/mode (mutually exclusive)
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "-d",
        "--dtmf",
        action="store_true",
        help="Use DTMF encoding (dual-tone, 16 symbols, ~40 bps)",
    )
    mode_group.add_argument(
        "-f",
        "--fsk",
        action="store_true",
        help="Use FSK encoding (single-tone, 400-2000Hz, ~80 bps)",
    )
    mode_group.add_argument(
        "-u",
        "--ultrasonic",
        action="store_true",
        help="Use ultrasonic encoding (15-20kHz, inaudible, ~1600 bps)",
    )
    mode_group.add_argument(
        "--tts",
        action="store_true",
        help="Use ElevenLabs TTS instead of frequency encoding",
    )
    mode_group.add_argument(
        "-b",
        "--benchmark",
        action="store_true",
        help="Run benchmark comparing all encoding methods",
    )

    # Optional
    parser.add_argument(
        "-bt",
        "--buffer-tokens",
        type=int,
        default=1,
        metavar="N",
        help="Number of tokens to buffer before encoding (default: 1)",
    )
    parser.add_argument(
        "--tone-duration",
        type=int,
        default=100,
        metavar="MS",
        help="Duration of each tone in ms (default: 100, ultrasonic default: 5)",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default=None,
        help="Specific model to use (default: gpt-4o-mini / claude-sonnet-4-5-20250929)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show detailed output (tokens, frequencies, timing)",
    )

    # Benchmark-specific options
    parser.add_argument(
        "-t",
        "--benchmark-text",
        type=str,
        help="Direct text to benchmark (skips LLM)",
    )
    parser.add_argument(
        "--no-tts",
        action="store_true",
        help="Skip TTS in benchmark",
    )
    parser.add_argument(
        "--play-audio",
        action="store_true",
        help="Actually play audio in benchmark (slower)",
    )

    args = parser.parse_args()

    # Handle benchmark mode
    if args.benchmark:
        if not args.prompt and not args.benchmark_text:
            args.benchmark_text = "The quick brown fox jumps over the lazy dog. " * 3
        run_benchmark(args)
        return

    # Validate args for streaming mode
    if not args.prompt:
        parser.error("--prompt is required for streaming mode")

    if not args.openai and not args.anthropic:
        parser.error("--openai or --anthropic is required")

    if not any([args.dtmf, args.fsk, args.ultrasonic, args.tts]):
        parser.error("--dtmf, --fsk, --ultrasonic, or --tts is required")

    asyncio.run(run_stream(args))


if __name__ == "__main__":
    main()
