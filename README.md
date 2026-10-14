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

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-WebSocket-009688?logo=fastapi&logoColor=white)
![Whisper](https://img.shields.io/badge/ASR-Whisper_small-orange?logo=openai&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-ready-2496ED?logo=docker&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

An end-to-end pipeline that listens to spoken Hindi, transcribes it, translates it, and speaks back the English — streamed live over WebSocket with word-level alignment and a full synthetic dataset generation pipeline.

---

## What It Does

```
Hindi speech (mic / file)
    │
    ▼  WebSocket  (80 ms PCM frames)
FastAPI server
    ├─► Whisper small      — ASR + word-level timestamps
    ├─► MarianMT (hi→en)   — neural machine translation
    └─► Edge TTS           — English speech synthesis
    │
    ▼  WebSocket  (≤1 s PCM frames)
Browser / CLI — plays English audio + shows transcript
```

---

## Demo

Open the web app, hit the mic button, speak in Hindi:

| Hindi input | English output |
|---|---|
| मुझे दिल्ली से मुंबई जाना है | I need to go from Delhi to Mumbai |
| डॉक्टर ने कहा कि रोज़ सुबह दवाई लेनी है | The doctor said to take medicine every morning |
| मेरे account में पचास हजार रुपये transfer कर दो | Transfer fifty thousand rupees to my account |

---

## Architecture

```
┌─────────────────┐   80 ms int16 PCM frames   ┌──────────────────────────────────┐
│  Browser /      │ ──── WebSocket ──────────► │          FastAPI Server          │
│  CLI client     │                             │                                  │
│                 │ ◄─── WebSocket ──────────  │  1. Whisper small                │
│  Plays English  │   English PCM (≤1 s/frame) │     ASR + word timestamps (DTW)  │
│  Shows Hindi    │                             │                                  │
│  transcript +   │                             │  2. Helsinki-NLP/opus-mt-hi-en   │
│  English text   │                             │     MarianMT  beam=4             │
└─────────────────┘                             │                                  │
                                                │  3. Edge TTS en-IN-NeerjaNeural  │
                                                │     English speech synthesis     │
                                                │                                  │
                                                │  4. Stream PCM back in chunks    │
                                                └──────────────────────────────────┘
```

---

## Project Structure

```
├── data_pipeline/
│   ├── run_pipeline.py       # Orchestrator — runs all 5 stages end-to-end
│   ├── audio_processor.py    # Load / resample / PCM ↔ float32 conversion
│   ├── transcriber.py        # Whisper ASR + word-level timestamp extraction
│   ├── translator.py         # MarianMT Hindi→English (batch + alternatives)
│   ├── tts_generator.py      # Edge TTS synthesis (Hindi & English voices)
│   ├── quality_filter.py     # Dataset record validation (duration, rate, timings)
│   └── manifest_builder.py   # JSONL manifest writer + stats
├── streaming_server/
│   ├── server.py             # FastAPI WebSocket inference server
│   └── client_demo.py        # CLI reference client (file or mic input)
├── webapp/
│   ├── app.py                # FastAPI server with extended WebSocket protocol
│   └── index.html            # Browser UI — Web Audio API, waveform, dark mode
├── sample_data/
│   └── hindi_sentences.txt   # 60 Hindi sentences across 10 domains
├── Dockerfile                # Pre-bakes models; cold-start-free
├── requirements.txt
└── LICENSE
```

---

## Components

### Streaming Server (`streaming_server/server.py`)

FastAPI WebSocket server for real-time inference.

| Parameter | Value |
|---|---|
| Endpoint | `ws://localhost:8000/ws` |
| Health check | `GET /health` |
| Frame size | 80 ms · 1920 samples · 3840 bytes |
| Max output frame | 1 second of English PCM |
| Inference | Async — thread pool executor, never blocks event loop |
| Startup | All models pre-warmed on boot |

**WebSocket protocol:**
```
client → session_start    {"type":"session_start","session_id":"s-001","sample_rate":24000,"encoding":"pcm_s16le","source_language":"hi","target_language":"en"}
client → audio frames     binary int16 PCM, 80 ms each, sent at real-time pace
client → input_end        {"type":"input_end","session_id":"s-001","valid_samples":N}
server → audio frames     binary int16 PCM, ≤1 s per frame
server → session_complete {"type":"session_complete","session_id":"s-001"}
```

---

### Web App (`webapp/`)

Full browser demo — no frameworks, pure HTML/CSS/JS.

- Mic capture via **Web Audio API** (`getUserMedia` → `ScriptProcessor`, 24 kHz)
- Live **waveform visualisation** using `AnalyserNode` + Canvas
- Sends 80 ms PCM frames over WebSocket at real-time pace
- Receives binary PCM + JSON `transcript` / `translation` messages simultaneously
- Reconstructs and **auto-plays** English audio in browser
- One-click **WAV download** (RIFF header built client-side)
- Shows round-trip latency, audio duration, session ID

---

### Data Pipeline (`data_pipeline/`)

Generates a labelled Hindi speech dataset in JSONL format.

**5 pipeline stages:**

| Stage | Tool | Output |
|---|---|---|
| Hindi TTS | Edge TTS `hi-IN-MadhurNeural` | 24 kHz mono WAV |
| ASR + word alignment | Whisper small `word_timestamps=True` | Transcript + per-word timings (DTW) |
| Translation | MarianMT `opus-mt-hi-en` beam=4 | English text + up to 2 alternatives |
| Quality filter | Duration · word count · char-rate | Pass / reject with issue list |
| Manifest | JSONL one record per line | Labelled dataset + stats |

**Quality rules:**
- Clip duration: 2–60 seconds
- Minimum 2 words
- Speech rate ≤ 35 chars/sec (catches TTS word-skipping artifacts)
- Word timings: non-negative, non-overlapping, within clip bounds

**JSONL record format:**
```json
{
  "id": "rec-000000",
  "audio": "audio/rec-000000.wav",
  "duration": 4.488,
  "speaker_id": "spk-edge-male",
  "source": {
    "corpus": "synthetic-edge-tts",
    "synthetic": true,
    "voice": "hi-IN-MadhurNeural"
  },
  "sentences": [{
    "hi_text": "मैंने कल अपना पुराना बैंक खाता बंद करवा दिया",
    "hi_start": 0.0,
    "hi_end": 3.5,
    "en_text": "I closed my old bank account yesterday",
    "en_alt": ["Yesterday I closed my old bank account"]
  }],
  "hi_words": [
    {"w": "मैंने", "start": 0.0,  "end": 0.62},
    {"w": "कल",   "start": 0.62, "end": 0.84},
    {"w": "अपना", "start": 0.84, "end": 1.3}
  ],
  "translation_method": "Helsinki-NLP/opus-mt-hi-en (MarianMT, beam=4)"
}
```

---

## Models

| Component | Model | Size | Notes |
|---|---|---|---|
| ASR | `openai/whisper-small` | ~244 MB | Word-level timestamps via DTW attention alignment |
| Translation | `Helsinki-NLP/opus-mt-hi-en` | ~300 MB | MarianMT, CPU-only, no GPU required |
| TTS | Edge TTS `en-IN-NeerjaNeural` | — | Free, no API key, Indian English accent |

---

## Audio Spec

| Parameter | Value |
|---|---|
| Sample rate | 24 kHz |
| Bit depth | 16-bit PCM (int16) |
| Channels | Mono |
| Frame duration | 80 ms = 1920 samples = 3840 bytes |

---

## Latency

| Hardware | Round-trip |
|---|---|
| CPU | ~25–35 s |
| GPU (H100) | ~2–3 s |

Bottleneck on CPU is Whisper decoding (~15 s). For faster inference swap to `whisper-large-v3` on GPU.

---

## Local Setup

```bash
# 1. Install PyTorch (CPU build)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

# 2. Install project dependencies
pip install -r requirements.txt

# 3. Start the web app
python webapp/app.py
# → Open http://localhost:8000
```

**Run the dataset pipeline:**
```bash
python data_pipeline/run_pipeline.py \
  --input-file sample_data/hindi_sentences.txt \
  --output-dir dataset_output \
  --num-samples 20
```

**Test the server from CLI (no browser needed):**
```bash
# From an audio file
python streaming_server/client_demo.py --input your_audio.wav --output english_out.wav

# From microphone
python streaming_server/client_demo.py --mic --duration 8
```

---

## Docker

```bash
docker build -t hindi-s2s .
docker run -p 7860:7860 hindi-s2s
# → Open http://localhost:7860
```

Models are downloaded and baked into the image at build time — the container starts with zero cold-start delay.

---

## Sample Data

`sample_data/hindi_sentences.txt` contains 60 Hindi sentences across 10 domains:

| Domain | Examples |
|---|---|
| Banking & finance | Account closures, transfers, loan queries |
| Healthcare | Doctor prescriptions, medical conditions |
| Travel & transport | Flight bookings, taxi, ride-hailing |
| Hinglish | Code-switched Hindi-English sentences |
| Negations | Multiple negatives in a single sentence |
| Self-corrections | Hesitations, mid-sentence word replacements |
| Names & places | Proper nouns, cities, Bollywood references |
| Numbers & dates | Phone numbers, amounts, day/month/year |
| Conditions | If-then logical structures |
| Products & e-commerce | Warranty, shipping, returns |

---

## Upgrade Path

| Component | Current | Production upgrade |
|---|---|---|
| ASR | Whisper small | WhisperX large-v3 (phoneme-level alignment, faster) |
| Translation | MarianMT | IndicTrans2 (better on Hinglish and code-switching) |
| TTS | Edge TTS | Bulbul v3 (natural Indian accent) |
| Architecture | Cascade (3 models) | Single end-to-end speech-to-speech checkpoint |

---

## License

[MIT](LICENSE)
