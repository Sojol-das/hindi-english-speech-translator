#!/usr/bin/env python3
"""
Streaming Hindi→English speech-to-speech WebSocket server.

WebSocket protocol for real-time Hindi→English speech-to-speech translation:

  Client → Server
    session_start  (JSON)   {"type":"session_start","session_id":"s-0001","sample_rate":24000,...}
    audio          (binary) raw PCM int16, 80 ms / frame = 3840 bytes
    input_end      (JSON)   {"type":"input_end","session_id":"s-0001","valid_samples":N}
    silence        (binary) continues until session_complete received

  Server → Client
    audio          (binary) English PCM int16, ≤ 1 s per frame
    session_complete (JSON) {"type":"session_complete","session_id":"s-0001"}

Entrypoint: port 8000, path /ws
Health:     GET /health → HTTP 200

Run:
  python streaming_server/server.py
"""

import asyncio
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

sys.path.insert(0, str(Path(__file__).parent.parent))

# ── audio constants ───────────────────────────────────────────────────────────
TARGET_SR = 24000
FRAME_MS = 80
FRAME_SAMPLES = int(TARGET_SR * FRAME_MS / 1000)   # 1920 samples
FRAME_BYTES = FRAME_SAMPLES * 2                     # int16 → 3840 bytes
MAX_FRAME_OUTPUT_SAMPLES = TARGET_SR                # ≤ 1 s per output frame
SESSION_COMPLETE_TIMEOUT = 30                        # seconds after input_end

# ── lazy model singletons ─────────────────────────────────────────────────────
_transcriber = None
_translator = None
_tts = None


def get_transcriber():
    global _transcriber
    if _transcriber is None:
        from data_pipeline.transcriber import HindiTranscriber
        _transcriber = HindiTranscriber(model_size="small")
    return _transcriber


def get_translator():
    global _translator
    if _translator is None:
        from data_pipeline.translator import HindiEnglishTranslator
        _translator = HindiEnglishTranslator()
    return _translator


def get_tts():
    global _tts
    if _tts is None:
        from data_pipeline.tts_generator import TTSGenerator
        _tts = TTSGenerator()
    return _tts


# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(title="Hindi→English S2S Server")


@app.on_event("startup")
async def _warm_models():
    """Pre-load models so the first session doesn't pay cold-start latency."""
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, get_transcriber)
    await loop.run_in_executor(None, get_translator)
    await loop.run_in_executor(None, get_tts)
    print("Models loaded and ready.")


@app.get("/health")
async def health():
    return JSONResponse({"status": "ok"})


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    session_id: str = "unknown"
    audio_buffer = bytearray()

    try:
        while True:
            msg = await asyncio.wait_for(
                ws.receive(),
                timeout=SESSION_COMPLETE_TIMEOUT + 15,
            )

            if "text" in msg:
                event = json.loads(msg["text"])
                etype = event.get("type")

                if etype == "session_start":
                    session_id = event.get("session_id", "s-?")
                    audio_buffer.clear()
                    print(f"[{session_id}] Started", flush=True)

                elif etype == "input_end":
                    valid_samples = event.get("valid_samples", len(audio_buffer) // 2)
                    print(f"[{session_id}] input_end — {valid_samples} samples "
                          f"({valid_samples / TARGET_SR:.2f}s)", flush=True)
                    await _process_session(ws, session_id, bytes(audio_buffer), valid_samples)
                    break   # session done

            elif "bytes" in msg:
                audio_buffer.extend(msg["bytes"])

    except asyncio.TimeoutError:
        print(f"[{session_id}] Timed out waiting for input_end")
    except WebSocketDisconnect:
        print(f"[{session_id}] Client disconnected")
    except Exception as exc:
        print(f"[{session_id}] Error: {exc}")
        raise


async def _process_session(
    ws: WebSocket,
    session_id: str,
    pcm_bytes: bytes,
    valid_samples: int,
):
    """
    Cascade pipeline:
      Hindi PCM → Whisper ASR → MarianMT translation → Edge TTS → English PCM
    """
    loop = asyncio.get_event_loop()

    # Trim to valid samples, convert to float32
    valid_bytes = pcm_bytes[: valid_samples * 2]
    audio = np.frombuffer(valid_bytes, dtype=np.int16).astype(np.float32) / 32768.0

    # 1. Transcribe (blocking → thread pool)
    tx = await loop.run_in_executor(None, lambda: get_transcriber().transcribe_array(audio))
    print(f"[{session_id}] ASR: {tx.text!r}", flush=True)

    if not tx.text.strip():
        await ws.send_json({"type": "session_complete", "session_id": session_id})
        return

    # 2. Translate
    en_text = await loop.run_in_executor(None, lambda: get_translator().translate(tx.text))
    print(f"[{session_id}] TL:  {en_text!r}", flush=True)

    # 3. English TTS → float32 array
    en_audio = await loop.run_in_executor(
        None, lambda: get_tts().generate_english_array(en_text)
    )

    # 4. Stream English PCM back in ≤ 1-second frames
    en_pcm = (en_audio.clip(-1.0, 1.0) * 32767).astype(np.int16).tobytes()
    chunk = MAX_FRAME_OUTPUT_SAMPLES * 2  # bytes
    for offset in range(0, len(en_pcm), chunk):
        await ws.send_bytes(en_pcm[offset : offset + chunk])

    await ws.send_json({"type": "session_complete", "session_id": session_id})
    print(f"[{session_id}] Done", flush=True)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
