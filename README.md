---
title: Hindi English Live Speech Translator
emoji: 🎙️
colorFrom: red
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
short_description: Real-time Hindi → English speech-to-speech translation demo
---

# Hindi → English Real-time Speech-to-Speech Translation

A personal project for real-time Hindi → English speech-to-speech translation.

## How to use

1. Click the red mic button and speak in Hindi
2. Click again to stop
3. Wait ~25–35 s for the translation (CPU inference)
4. Hear the English translation played back automatically

Try: _"मुझे दिल्ली से मुंबई जाना है"_ or _"मेरे account में पचास हजार रुपये transfer कर दो"_

## Architecture

```
Browser mic → WebSocket → FastAPI server
    → Whisper small (ASR)
    → Helsinki-NLP/opus-mt-hi-en (translation)
    → Edge TTS en-IN (English speech)
    → WebSocket back → browser plays audio
```

| Component | Model | Notes |
|---|---|---|
| ASR | `openai/whisper-small` | Word-level timestamps |
| Translation | `Helsinki-NLP/opus-mt-hi-en` | MarianMT, CPU-friendly |
| TTS | Edge TTS `en-IN-NeerjaNeural` | Upgrade: Bulbul v3 |

## Dataset pipeline

The `data_pipeline/` module generates training data in JSONL format:
- Hindi TTS → Whisper transcription + word alignment → MarianMT translation → manifest

## Latency note

Running on CPU, round-trip is ~25–35 s. On an H100 this drops to ~2–3 s.

## Local run

```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install openai-whisper transformers sentencepiece sacremoses edge-tts librosa soundfile fastapi "uvicorn[standard]" websockets tqdm

PORT=8000 python webapp/app.py
# open http://localhost:8000
```
