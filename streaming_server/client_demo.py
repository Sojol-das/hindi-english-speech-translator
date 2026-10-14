#!/usr/bin/env python3
"""
Demo client for the Hindi→English S2S streaming server.

Implements the client side of the WebSocket streaming protocol:
  - Streams Hindi audio in 80 ms PCM frames in real time
  - Receives English PCM frames and plays them back

Usage:
  python streaming_server/client_demo.py --input audio.wav
  python streaming_server/client_demo.py --mic --duration 8
  python streaming_server/client_demo.py --input audio.wav --output english_out.wav
"""

import argparse
import asyncio
import json
import sys
import time

import numpy as np
import sounddevice as sd
import soundfile as sf
import websockets

TARGET_SR = 24000
FRAME_MS = 80
FRAME_SAMPLES = int(TARGET_SR * FRAME_MS / 1000)   # 1920
FRAME_BYTES = FRAME_SAMPLES * 2                     # int16

SERVER_URL = "ws://localhost:8000/ws"


async def run_session(audio: np.ndarray, session_id: str = "s-0001") -> np.ndarray:
    """
    Send Hindi audio to the server following the WebSocket protocol.
    Returns the received English audio as a float32 array.
    """
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    pcm_bytes = pcm.tobytes()
    total_samples = len(pcm)
    total_frames = (len(pcm_bytes) + FRAME_BYTES - 1) // FRAME_BYTES

    received: bytearray = bytearray()
    session_done = asyncio.Event()

    async with websockets.connect(SERVER_URL, max_size=None) as ws:

        # ── receive task ──────────────────────────────────────────────────────
        async def receiver():
            async for message in ws:
                if isinstance(message, bytes):
                    received.extend(message)
                elif isinstance(message, str):
                    event = json.loads(message)
                    if event.get("type") == "session_complete":
                        session_done.set()
                        break

        recv_task = asyncio.create_task(receiver())

        # ── send session_start ────────────────────────────────────────────────
        await ws.send(json.dumps({
            "type": "session_start",
            "session_id": session_id,
            "sample_rate": TARGET_SR,
            "encoding": "pcm_s16le",
            "source_language": "hi",
            "target_language": "en",
        }))

        # ── stream audio frames in real time ──────────────────────────────────
        t0 = time.monotonic()
        for k in range(total_frames):
            frame = pcm_bytes[k * FRAME_BYTES : (k + 1) * FRAME_BYTES]
            if len(frame) < FRAME_BYTES:
                frame = frame + b"\x00" * (FRAME_BYTES - len(frame))
            await ws.send(frame)

            # Throttle to real-time: frame k should go out at t = k * 80ms
            deadline = t0 + (k + 1) * (FRAME_MS / 1000)
            sleep = deadline - time.monotonic()
            if sleep > 0:
                await asyncio.sleep(sleep)

        # ── send input_end ────────────────────────────────────────────────────
        await ws.send(json.dumps({
            "type": "input_end",
            "session_id": session_id,
            "valid_samples": total_samples,
        }))

        # ── wait for session_complete (up to 30s) ────────────────────────────
        try:
            await asyncio.wait_for(session_done.wait(), timeout=35)
        except asyncio.TimeoutError:
            print("Warning: session_complete not received within 35 s")

        recv_task.cancel()

    if not received:
        return np.array([], dtype=np.float32)

    en_pcm = np.frombuffer(bytes(received), dtype=np.int16).astype(np.float32) / 32768.0
    return en_pcm


def record_mic(duration: float) -> np.ndarray:
    print(f"Recording {duration:.0f}s — speak Hindi now...")
    audio = sd.rec(int(duration * TARGET_SR), samplerate=TARGET_SR, channels=1, dtype="float32")
    sd.wait()
    print("Recording done.")
    return audio.flatten()


async def main(args):
    if args.mic:
        audio = record_mic(args.duration)
    else:
        audio, sr = sf.read(args.input, dtype="float32")
        if sr != TARGET_SR:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        print(f"Loaded {args.input}  ({len(audio)/TARGET_SR:.2f}s)")

    print("Sending to server...")
    t0 = time.monotonic()
    en_audio = await run_session(audio)
    elapsed = time.monotonic() - t0
    print(f"Received {len(en_audio)/TARGET_SR:.2f}s of English audio  (round-trip: {elapsed:.1f}s)")

    if len(en_audio) == 0:
        print("No audio returned from server.")
        return

    print("Playing English translation...")
    sd.play(en_audio, TARGET_SR)
    sd.wait()

    if args.output:
        sf.write(args.output, en_audio, TARGET_SR, subtype="PCM_16")
        print(f"Saved to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hindi→English S2S demo client")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--input", help="Hindi WAV/MP3 file to translate")
    src.add_argument("--mic", action="store_true", help="Record from microphone")
    parser.add_argument("--duration", type=float, default=10.0, help="Mic recording duration (s)")
    parser.add_argument("--output", help="Save English audio to this WAV file")
    parser.add_argument("--session-id", default="s-0001")
    args = parser.parse_args()

    asyncio.run(main(args))
