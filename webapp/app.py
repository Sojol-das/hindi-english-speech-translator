#!/usr/bin/env python3
"""
Web app demo for Hindi → English real-time speech-to-speech translation.

Extends the WebSocket streaming protocol with two extra server→client JSON messages:
  {"type": "transcript",  "text": "<Hindi ASR result>"}
  {"type": "translation", "text": "<English translation>"}

Run:
  python webapp/app.py
  Open: http://localhost:8000
"""

import asyncio
import json
import sys
from pathlib import Path

import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse

sys.path.insert(0, str(Path(__file__).parent.parent))

TARGET_SR = 24000
MAX_OUTPUT_SAMPLES = TARGET_SR      # ≤ 1 s per output frame
SESSION_TIMEOUT = 60                # generous for demo

# ── lazy singletons ──────────────────────────────────────────────────────────
_transcriber = _translator = _tts = None

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

# ── app ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="Hindi→English S2S Demo")

@app.on_event("startup")
async def warmup():
    loop = asyncio.get_event_loop()
    print("Loading models...", flush=True)
    await loop.run_in_executor(None, get_transcriber)
    await loop.run_in_executor(None, get_translator)
    await loop.run_in_executor(None, get_tts)
    print("Models ready.", flush=True)

@app.get("/")
async def root():
    html_path = Path(__file__).parent / "index.html"
    return HTMLResponse(html_path.read_text(encoding="utf-8"))

@app.get("/health")
async def health():
    return JSONResponse({"status": "ok"})

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    session_id = "unknown"
    audio_buffer = bytearray()

    try:
        while True:
            msg = await asyncio.wait_for(ws.receive(), timeout=SESSION_TIMEOUT)

            if "text" in msg:
                event = json.loads(msg["text"])
                etype = event.get("type")

                if etype == "session_start":
                    session_id = event.get("session_id", "s-?")
                    audio_buffer.clear()
                    print(f"[{session_id}] started", flush=True)

                elif etype == "input_end":
                    valid_samples = event.get("valid_samples", len(audio_buffer) // 2)
                    print(f"[{session_id}] input_end  {valid_samples/TARGET_SR:.2f}s", flush=True)
                    await _process(ws, session_id, bytes(audio_buffer), valid_samples)
                    break

            elif "bytes" in msg:
                audio_buffer.extend(msg["bytes"])

    except asyncio.TimeoutError:
        print(f"[{session_id}] timeout", flush=True)
    except WebSocketDisconnect:
        print(f"[{session_id}] disconnected", flush=True)
    except Exception as e:
        print(f"[{session_id}] error: {e}", flush=True)


async def _process(ws: WebSocket, session_id: str, pcm_bytes: bytes, valid_samples: int):
    loop = asyncio.get_event_loop()

    audio = np.frombuffer(pcm_bytes[: valid_samples * 2], dtype=np.int16).astype(np.float32) / 32768.0

    # 1. ASR
    tx = await loop.run_in_executor(None, lambda: get_transcriber().transcribe_array(audio))
    print(f"[{session_id}] ASR: {tx.text!r}", flush=True)
    await ws.send_json({"type": "transcript", "text": tx.text})

    if not tx.text.strip():
        await ws.send_json({"type": "session_complete", "session_id": session_id})
        return

    # 2. Translation
    en_text = await loop.run_in_executor(None, lambda: get_translator().translate(tx.text))
    print(f"[{session_id}] TL:  {en_text!r}", flush=True)
    await ws.send_json({"type": "translation", "text": en_text})

    # 3. TTS → English audio
    en_audio = await loop.run_in_executor(None, lambda: get_tts().generate_english_array(en_text))

    # 4. Stream back in ≤ 1-second PCM frames
    pcm_out = (np.clip(en_audio, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
    chunk = MAX_OUTPUT_SAMPLES * 2
    for offset in range(0, len(pcm_out), chunk):
        await ws.send_bytes(pcm_out[offset : offset + chunk])

    await ws.send_json({"type": "session_complete", "session_id": session_id})
    print(f"[{session_id}] done", flush=True)


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
