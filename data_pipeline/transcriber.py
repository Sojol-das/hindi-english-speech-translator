"""
Hindi ASR with word-level timestamps via OpenAI Whisper.

Whisper's word_timestamps=True gives approximate word boundaries using
DTW attention alignment. For higher-quality phoneme-level alignment,
swap in WhisperX (optional; requires ctc-forced-aligner).
"""

import librosa
import numpy as np
import whisper
from dataclasses import dataclass, field
from typing import List

TARGET_SR = 24000
WHISPER_SR = 16000  # Whisper always expects 16 kHz internally


@dataclass
class WordAlignment:
    word: str
    start: float  # seconds from clip start
    end: float


@dataclass
class TranscriptionResult:
    text: str
    words: List[WordAlignment] = field(default_factory=list)
    language: str = "hi"


class HindiTranscriber:
    def __init__(self, model_size: str = "small"):
        print(f"Loading Whisper '{model_size}' model...")
        self.model = whisper.load_model(model_size)

    def transcribe(self, audio_path: str) -> TranscriptionResult:
        """Transcribe a WAV file, returning verbatim text + word timings.
        Loads via librosa to avoid requiring ffmpeg on the system."""
        audio_16k, _ = librosa.load(audio_path, sr=WHISPER_SR, mono=True)
        return self.transcribe_array(audio_16k, already_16k=True)

    def transcribe_array(self, audio: np.ndarray, already_16k: bool = False) -> TranscriptionResult:
        """Transcribe a float32 audio array.
        If already_16k=False (default), assumes 24 kHz and resamples."""
        if not already_16k:
            audio = librosa.resample(audio, orig_sr=TARGET_SR, target_sr=WHISPER_SR)
        result = self.model.transcribe(
            audio,
            language="hi",
            word_timestamps=True,
            verbose=False,
            condition_on_previous_text=False,
        )
        return self._parse_result(result)

    def _parse_result(self, result: dict) -> TranscriptionResult:
        words: List[WordAlignment] = []
        for segment in result["segments"]:
            for w in segment.get("words", []):
                text = w["word"].strip()
                if text:
                    words.append(
                        WordAlignment(
                            word=text,
                            start=round(max(0.0, w["start"]), 3),
                            end=round(max(0.0, w["end"]), 3),
                        )
                    )
        return TranscriptionResult(
            text=result["text"].strip(),
            words=words,
            language=result.get("language", "hi"),
        )
