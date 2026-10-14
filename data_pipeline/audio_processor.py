"""Audio loading, resampling, segmentation, and PCM conversion utilities."""

import librosa
import numpy as np
import soundfile as sf
from pathlib import Path

TARGET_SR = 24000  # mono 16-bit PCM at 24 kHz


class AudioProcessor:
    def load(self, path: str) -> np.ndarray:
        """Load any audio file, resample to 24 kHz mono float32."""
        audio, _ = librosa.load(path, sr=TARGET_SR, mono=True)
        return audio

    def save(self, audio: np.ndarray, path: str):
        """Save float32 audio as 16-bit WAV at 24 kHz."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        sf.write(path, audio, TARGET_SR, subtype="PCM_16")

    def duration(self, audio: np.ndarray) -> float:
        return len(audio) / TARGET_SR

    def segment(self, audio: np.ndarray, start: float, end: float) -> np.ndarray:
        return audio[int(start * TARGET_SR) : int(end * TARGET_SR)]

    def pcm_bytes_to_array(self, pcm_bytes: bytes) -> np.ndarray:
        """Convert raw int16 PCM bytes to float32 [-1, 1]."""
        return np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0

    def array_to_pcm_bytes(self, audio: np.ndarray) -> bytes:
        """Convert float32 [-1, 1] audio to raw int16 PCM bytes."""
        clipped = np.clip(audio, -1.0, 1.0)
        return (clipped * 32767).astype(np.int16).tobytes()

    def file_duration(self, path: str) -> float:
        return librosa.get_duration(path=path)
