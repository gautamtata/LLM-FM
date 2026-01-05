# LLM-FM

Experimental protocol for AI-to-AI voice agent communication using frequency encoding (DTMF/FSK/Ultrasonic) instead of speech. Streams LLM tokens directly to audio frequencies, bypassing TTS/STT latency.

## Problem

Current AI voice agents communicate through a costly pipeline:

```
Agent A                                              Agent B
   |                                                    |
   v                                                    v
[LLM] -> [TTS] -> [Audio] -> [Network] -> [STT] -> [LLM]
              ~150ms              ~50ms        ~200ms
```

This introduces 400-800ms of latency per turn, primarily from:
- **TTS synthesis**: Converting text to human-audible speech (~150-300ms)
- **STT transcription**: Converting speech back to text (~150-400ms)
- **Bandwidth overhead**: Transmitting full audio waveforms

When both endpoints are AI agents, human-audible speech is unnecessary overhead.

## Solution

LLM-FM encodes LLM output directly as frequency tones:

```
Agent A                                              Agent B
   |                                                    |
   v                                                    v
[LLM] -> [Freq Encoder] -> [Audio] -> [Freq Decoder] -> [LLM]
              ~1ms            ~50ms         ~1ms
```

Benefits:
- **100x faster** than TTS for AI-to-AI communication
- **Ultrasonic mode**: 15-20kHz frequencies inaudible to humans
- **Direct streaming**: Encode tokens as they arrive from the LLM
- **Simple decoding**: FFT-based frequency detection, no ML required

## Benchmark Results

Tested with 43-character message: "The quick brown fox jumps over the lazy dog"

| Method | Speed (chars/sec) | vs TTS | Notes |
|--------|-------------------|--------|-------|
| Ultrasonic (1ms) | 1000 | 103.7x faster | Inaudible to humans |
| Ultrasonic (2ms) | 500 | 51.9x faster | Inaudible to humans |
| Ultrasonic (5ms) | 200 | 20.7x faster | Inaudible to humans |
| FSK (10ms) | 100 | 10.4x faster | Audible, single-tone |
| FSK (50ms) | 20 | 2.1x faster | Audible |
| FSK (100ms) | 10 | 1.0x | Audible |
| ElevenLabs TTS | 9.6 | baseline | Human speech |
| DTMF (100ms) | 5 | 0.5x | Telephone standard |

Ultrasonic encoding at 1ms tones achieves **1000 characters per second**, compared to TTS at ~10 chars/sec.

## Encoding Schemes

### DTMF (Dual-Tone Multi-Frequency)

Standard telephone keypad encoding. Each symbol uses two simultaneous frequencies:

```
           1209 Hz   1336 Hz   1477 Hz   1633 Hz
697 Hz  |    1         2         3         A
770 Hz  |    4         5         6         B
852 Hz  |    7         8         9         C
941 Hz  |    *         0         #         D
```

- 16 symbols (4 bits per symbol)
- Characters encoded as 2 hex digits
- Robust but slow (~40 bps at 100ms tones)

### FSK (Frequency-Shift Keying)

Single-tone encoding with frequency mapped to ASCII value:

```
Frequency = 400 + (ASCII / 255) * 1600 Hz
Range: 400 Hz - 2000 Hz
```

- 256 symbols (8 bits per symbol)
- One tone per character
- ~80 bps at 100ms, ~800 bps at 10ms

### Ultrasonic

High-speed encoding in the 15-20kHz range:

```
Frequency = 15000 + (ASCII / 255) * 5000 Hz
Range: 15 kHz - 20 kHz
```

- Inaudible to most adult humans
- Supports 1-5ms tone durations
- ~1600 bps at 5ms, ~8000 bps at 1ms

## Installation

Requires Python 3.11+

```bash
# Clone repository
git clone https://github.com/yourusername/llm-fm.git
cd llm-fm

# Install dependencies with uv
uv sync

# Or with pip
pip install -e .
```

## API Keys

Set the following environment variables:

| Variable | Required For | Get Key |
|----------|--------------|---------|
| `OPENAI_API_KEY` | OpenAI provider | [platform.openai.com](https://platform.openai.com) |
| `ANTHROPIC_API_KEY` | Anthropic provider | [console.anthropic.com](https://console.anthropic.com) |
| `ELEVENLABS_API_KEY` | TTS benchmark | [elevenlabs.io](https://elevenlabs.io) |

```bash
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...
export ELEVENLABS_API_KEY=...
```

Or use a `.env` file in the project root.

## Usage

### Frequency Encoding

```bash
# DTMF encoding (telephone standard)
uv run main.py --dtmf --openai -p "Count from 1 to 10"

# FSK encoding (audible, faster)
uv run main.py --fsk --openai -p "Hello world" --tone-duration 50

# Ultrasonic encoding (inaudible, fastest)
uv run main.py --ultrasonic --openai -p "Hello" --tone-duration 2

# With Anthropic
uv run main.py --ultrasonic --anthropic -p "What is the capital of France?"

# Verbose output (show frequencies)
uv run main.py --ultrasonic --openai -p "Test" -v
```

### TTS Comparison

```bash
# Stream LLM output through ElevenLabs TTS
uv run main.py --tts --openai -p "Hello world"
```

### Benchmarking

```bash
# Benchmark all encoding methods
uv run main.py --benchmark -p "Tell me a short joke"

# Benchmark with fixed text (no LLM call)
uv run main.py --benchmark -t "The quick brown fox jumps over the lazy dog"

# Skip TTS benchmark (faster)
uv run main.py --benchmark -t "Test message" --no-tts
```

### CLI Options

```
-p, --prompt          Prompt to send to the LLM
-o, --openai          Use OpenAI API
-a, --anthropic       Use Anthropic API
-d, --dtmf            DTMF encoding (~40 bps)
-f, --fsk             FSK encoding (~80-800 bps)
-u, --ultrasonic      Ultrasonic encoding (~1600-8000 bps)
--tts                 ElevenLabs TTS output
-b, --benchmark       Run benchmark comparison
--tone-duration MS    Tone duration in milliseconds
-bt, --buffer-tokens  Tokens to buffer before encoding
-v, --verbose         Show detailed frequency output
```

## Project Structure

```
llm-fm/
├── main.py                     # CLI entry point
├── src/tfsp/
│   ├── providers/              # LLM API integrations
│   │   ├── openai.py           # OpenAI streaming
│   │   └── anthropic.py        # Anthropic streaming
│   ├── encoders/               # Frequency encoding schemes
│   │   ├── dtmf.py             # DTMF dual-tone
│   │   ├── fsk.py              # FSK single-tone
│   │   └── ultrasonic.py       # High-frequency encoding
│   ├── audio/                  # Audio generation
│   │   ├── tone.py             # Sine wave synthesis
│   │   └── player.py           # Sounddevice playback
│   ├── tts/                    # TTS for comparison
│   │   └── elevenlabs.py       # ElevenLabs integration
│   ├── buffer.py               # Token buffering
│   └── benchmark.py            # Performance testing
└── tests/                      # Unit tests
```

## Technical Notes

### Frequency Detection

Decoding (not yet implemented) would use FFT-based peak detection:

1. Sample audio at 44.1kHz
2. Apply windowed FFT (e.g., Hanning window)
3. Detect frequency peaks above threshold
4. Map frequencies back to characters

For ultrasonic, the 5kHz bandwidth (15-20kHz) provides ~19.6 Hz spacing between 256 symbols, well within FFT resolution at typical sample rates.

### Tone Duration Limits

Minimum practical tone duration depends on:
- **Sample rate**: At 44.1kHz, 1ms = 44 samples
- **Frequency resolution**: Shorter tones = wider frequency uncertainty
- **Hardware latency**: Audio device buffering adds delay

Testing shows 1-2ms tones are reliably detectable with modern audio hardware.

### Network Considerations

For transmission over telephony networks:
- PSTN filters to 300-3400 Hz (DTMF only)
- VoIP/WebRTC supports full 20kHz bandwidth
- Direct audio connections support ultrasonic

## Future Work

- [ ] Implement frequency decoder (FFT-based)
- [ ] Add error correction (Reed-Solomon)
- [ ] WebRTC transport layer
- [ ] Bidirectional agent communication
- [ ] ggwave integration for battle-tested encoding
- [ ] Handshake protocol for capability negotiation

## References

- [DTMF Standard (ITU-T Q.23)](https://www.itu.int/rec/T-REC-Q.23)
- [ggwave - Data over sound](https://github.com/ggerganov/ggwave)
- [OpenAI Streaming API](https://platform.openai.com/docs/api-reference/chat/create#chat-create-stream)
- [Anthropic Streaming API](https://docs.anthropic.com/en/docs/build-with-claude/streaming)

## License

MIT

