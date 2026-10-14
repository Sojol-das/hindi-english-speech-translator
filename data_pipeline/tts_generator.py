"""
Text-to-speech using Microsoft Edge TTS (edge-tts).

Used for:
  - Generating synthetic Hindi audio for the dataset
  - Generating English target audio for streaming demo responses

English target audio can be upgraded to Bulbul:v3 voice for higher quality.
This module generates equivalent audio for local development.
"""

import asyncio
import tempfile
from pathlib import Path
from typing import Optional

import librosa
import numpy as np
import soundfile as sf

TARGET_SR = 24000

HINDI_VOICES = {
    "male": "hi-IN-MadhurNeural",
    "female": "hi-IN-SwaraNeural",
}
ENGLISH_VOICES = {
    "male": "en-IN-PrabhatNeural",
    "female": "en-IN-NeerjaNeural",
}


class TTSGenerator:
    def _run(self, coro):
        """Run an async coroutine from sync context safely."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    return pool.submit(asyncio.run, coro).result()
            return loop.run_until_complete(coro)
        except RuntimeError:
            return asyncio.run(coro)

    async def _speak(self, text: str, voice: str, output_mp3: str):
        import edge_tts
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(output_mp3)

    def _generate(self, text: str, voice: str, output_wav: str) -> float:
        """Generate TTS audio, convert to 24 kHz WAV, return duration in seconds."""
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            tmp = f.name
        try:
            self._run(self._speak(text, voice, tmp))
            audio, _ = librosa.load(tmp, sr=TARGET_SR, mono=True)
            Path(output_wav).parent.mkdir(parents=True, exist_ok=True)
            sf.write(output_wav, audio, TARGET_SR, subtype="PCM_16")
            return len(audio) / TARGET_SR
        finally:
            Path(tmp).unlink(missing_ok=True)

    def generate_hindi(
        self,
        text: str,
        output_path: str,
        voice: str = "male",
    ) -> float:
        """Generate Hindi speech. Returns duration in seconds."""
        return self._generate(text, HINDI_VOICES[voice], output_path)

    def generate_english(
        self,
        text: str,
        output_path: str,
        voice: str = "female",
    ) -> float:
        """Generate English speech. Returns duration in seconds."""
        return self._generate(text, ENGLISH_VOICES[voice], output_path)

    def generate_english_array(self, text: str, voice: str = "female") -> np.ndarray:
        """Generate English speech, return as float32 numpy array."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp = f.name
        try:
            self._generate(text, ENGLISH_VOICES[voice], tmp)
            audio, _ = librosa.load(tmp, sr=TARGET_SR, mono=True)
            return audio
        finally:
            Path(tmp).unlink(missing_ok=True)
