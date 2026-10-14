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

# Hindi → English Real-Time Speech-to-Speech Translation

A end-to-end pipeline that listens to spoken Hindi, transcribes it, translates it, and speaks back the English — all over a live WebSocket stream.

---

## Demo

1. Open the web app and click the red mic button
2. Speak in Hindi
3. Receive the English translation as synthesised audio with the transcript shown on screen

**Try these phrases:**
- _"मुझे दिल्ली से मुंबई जाना है"_ — travel intent
- _"मेरे account में पचास हजार रुपये transfer कर दो"_ — Hinglish banking
- _"डॉक्टर ने कहा कि रोज़ सुबह दवाई लेनी है"_ — healthcare

---

## Architecture

```
┌─────────────┐   80 ms PCM frames   ┌──────────────────────────────────────┐
│  Browser /  │ ──── WebSocket ────► │           FastAPI Server             │
│  CLI client │                      │                                      │
│             │ ◄─── WebSocket ────  │  Hindi PCM                           │
│  plays back │   English PCM frames │      │                               │
│  English    │                      │      ▼                               │
└─────────────┘                      │  Whisper small  (ASR, word timestamps)│
                                     │      │                               │
                                     │      ▼                               │
                                     │  MarianMT  (Helsinki-NLP hi→en)      │
                                     │      │                               │
                                     │      ▼                               │
                                     │  Edge TTS  (en-IN-NeerjaNeural)      │
                                     │      │                               │
                                     │      ▼                               │
                                     │  Stream ≤1 s frames back             │
                                     └──────────────────────────────────────┘
```

---

## Components

### `streaming_server/` — WebSocket inference server

| Detail | Value |
|---|---|
| Endpoint | `ws://localhost:8000/ws` |
| Frame size | 80 ms · 1920 samples · 3840 bytes (int16 PCM) |
| Max output per frame | 1 second of English PCM |
| Health check | `GET /health` |
| Concurrency | Async FastAPI; model inference runs in thread pool |
| Startup | Models pre-loaded on boot — zero cold-start on first request |

**Protocol flow:**
```
client  →  session_start (JSON)
client  →  audio frames  (binary, 80 ms each, real-time paced)
client  →  input_end     (JSON, with valid_samples count)
server  →  audio frames  (binary, English PCM, ≤1 s each)
server  →  session_complete (JSON)
```

### `webapp/` — Browser demo

- Records mic audio via **Web Audio API** (`ScriptProcessor`, 24 kHz)
- Live waveform visualisation using `AnalyserNode`
- Streams 80 ms PCM frames over WebSocket in real time
- Displays Hindi transcript and English translation as they arrive
- Plays back received English PCM audio; offers WAV download
- Dark-mode UI, no frameworks — pure HTML/CSS/JS

### `data_pipeline/` — Synthetic dataset generation

Generates a labelled Hindi speech dataset in JSONL format.

**Pipeline stages:**

| Stage | Tool | Output |
|---|---|---|
| Hindi speech synthesis | Edge TTS `hi-IN-MadhurNeural` | 24 kHz mono WAV |
| ASR + word alignment | Whisper small (`word_timestamps=True`) | Transcript + per-word timings |
| Translation | MarianMT `opus-mt-hi-en` (beam=4) | English text + alternatives |
| Quality filter | Duration · word count · char-rate checks | Pass / reject |
| Manifest | JSONL (one record per line) | Labelled dataset |

**JSONL record structure:**
```json
{
  "id": "rec-000001",
  "audio": "audio/rec-000001.wav",
  "duration": 4.32,
  "speaker_id": "spk-edge-male",
  "source": { "corpus": "synthetic-edge-tts", "synthetic": true, "voice": "hi-IN-MadhurNeural" },
  "sentences": [{ "hi_text": "...", "hi_start": 0.21, "hi_end": 4.10, "en_text": "...", "en_alt": ["..."] }],
  "hi_words": [{ "w": "मैंने", "start": 0.21, "end": 0.55 }],
  "translation_method": "Helsinki-NLP/opus-mt-hi-en (MarianMT, beam=4)"
}
```

**Quality filters applied:**
- Duration: 2–60 seconds
- Minimum 2 words
- Speech rate ≤ 35 chars/sec (catches TTS skipping artifacts)
- Word timings: non-negative, non-overlapping, within clip bounds

---

## Models

| Component | Model | Size | Notes |
|---|---|---|---|
| ASR | `openai/whisper-small` | ~244 MB | Word-level timestamps via DTW attention |
| Translation | `Helsinki-NLP/opus-mt-hi-en` | ~300 MB | MarianMT, CPU-only, beam search |
| TTS | Edge TTS `en-IN-NeerjaNeural` | — | Free, no API key required |

---

## Audio Spec

| Parameter | Value |
|---|---|
| Sample rate | 24 kHz |
| Bit depth | 16-bit PCM |
| Channels | Mono |
| Frame duration | 80 ms |

---

## Latency

| Hardware | Round-trip |
|---|---|
| CPU | ~25–35 s |
| H100 GPU | ~2–3 s |

Bottleneck on CPU is Whisper decoding (~15 s). Swap to `whisper-large-v3` + GPU for production.

---

## Local Setup

```bash
# 1. Install PyTorch (CPU)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the web app
python webapp/app.py
# Open http://localhost:8000
```

**Run dataset pipeline:**
```bash
python data_pipeline/run_pipeline.py \
  --input-file sample_data/hindi_sentences.txt \
  --output-dir dataset_output \
  --num-samples 20
```

**Run CLI client (test server without browser):**
```bash
# From a file
python streaming_server/client_demo.py --input your_audio.wav

# From mic
python streaming_server/client_demo.py --mic --duration 8
```

---

## Docker

```bash
docker build -t hindi-s2s .
docker run -p 7860:7860 hindi-s2s
```

Models are downloaded and baked in at build time — the container starts instantly with no cold-start downloads.

---

## Project Structure

```
├── data_pipeline/
│   ├── run_pipeline.py       # Orchestrator — runs all stages end-to-end
│   ├── audio_processor.py    # Load / resample / PCM conversion utilities
│   ├── transcriber.py        # Whisper ASR + word-level timestamp extraction
│   ├── translator.py         # MarianMT Hindi→English translation
│   ├── tts_generator.py      # Edge TTS speech synthesis (Hindi & English)
│   ├── quality_filter.py     # Dataset record validation
│   └── manifest_builder.py   # JSONL manifest writer
├── streaming_server/
│   ├── server.py             # FastAPI WebSocket inference server
│   └── client_demo.py        # CLI reference client
├── webapp/
│   ├── app.py                # FastAPI server + extended WebSocket protocol
│   └── index.html            # Browser UI (Web Audio API, vanilla JS)
├── sample_data/
│   └── hindi_sentences.txt   # 60 Hindi sentences across 10 domains
├── Dockerfile
└── requirements.txt
```

---

## Upgrade Path

| Component | Current | Production upgrade |
|---|---|---|
| ASR | Whisper small | WhisperX large-v3 (faster, phoneme alignment) |
| Translation | MarianMT | IndicTrans2 (better Hinglish handling) |
| TTS | Edge TTS | Bulbul v3 (higher quality, Indian accent) |
| Architecture | Cascade | Single end-to-end S2S checkpoint |
