"""ElevenLabs TTS implementation for baseline comparison."""

import io
import time
from dataclasses import dataclass

from elevenlabs import ElevenLabs
from elevenlabs.play import play


@dataclass
class TTSResult:
    """Result of TTS synthesis."""

    audio_data: bytes
    text: str
    synthesis_time_ms: float
    char_count: int

    @property
    def chars_per_second(self) -> float:
        """Calculate characters synthesized per second."""
        if self.synthesis_time_ms == 0:
            return 0
        return self.char_count / (self.synthesis_time_ms / 1000)


class ElevenLabsTTS:
    """ElevenLabs text-to-speech for baseline comparison."""

    def __init__(
        self,
        api_key: str | None = None,
        voice_id: str = "JBFqnCBsd6RMkjVDRZzb",  # George
        model_id: str = "eleven_multilingual_v2",
    ):
        """
        Initialize ElevenLabs TTS.

        Args:
            api_key: ElevenLabs API key (if None, uses ELEVENLABS_API_KEY env var)
            voice_id: Voice ID to use
            model_id: Model ID to use
        """
        self.client = ElevenLabs(api_key=api_key)
        self.voice_id = voice_id
        self.model_id = model_id

    def synthesize(self, text: str) -> TTSResult:
        """
        Synthesize text to speech.

        Args:
            text: Text to synthesize

        Returns:
            TTSResult with audio data and timing info
        """
        start_time = time.perf_counter()

        audio_generator = self.client.text_to_speech.convert(
            text=text,
            voice_id=self.voice_id,
            model_id=self.model_id,
            output_format="mp3_44100_128",
        )

        # Collect all audio chunks
        audio_data = b"".join(audio_generator)

        end_time = time.perf_counter()
        synthesis_time_ms = (end_time - start_time) * 1000

        return TTSResult(
            audio_data=audio_data,
            text=text,
            synthesis_time_ms=synthesis_time_ms,
            char_count=len(text),
        )

    def synthesize_and_play(self, text: str) -> TTSResult:
        """
        Synthesize text and play it.

        Args:
            text: Text to synthesize and play

        Returns:
            TTSResult with timing info
        """
        result = self.synthesize(text)
        # Play the audio
        play(io.BytesIO(result.audio_data))
        return result

