import asyncio
import audioop
import json
import os
import re
from contextlib import asynccontextmanager
from threading import Event, Thread

import numpy as np
import torch
import torchaudio
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from transformers import AutoModelForCausalLM, TextIteratorStreamer

from audiotokenizer import AudioTokenizerV2

MODEL_ID = os.environ.get("MODEL_ID", "saheedniyi/YarnGPT2")
WAV_CONFIG = os.environ.get("WAV_TOKENIZER_CONFIG", "/app/wavtokenizer.yaml")
WAV_MODEL = os.environ.get("WAV_TOKENIZER_MODEL", "/app/wavtokenizer.ckpt")

FIRST_CHUNK_TOKENS = 25  # small first chunk keeps time-to-first-audio low
CHUNK_TOKENS = 75        # then 1s chunks: fewer decode calls and boundaries
OVERLAP_TOKENS = 2
SAMPLES_PER_TOKEN = 320  # 24000 Hz / 75 tokens per second
TARGET_RMS = 0.125       # ~-18 dBFS; model output averages ~-24 dBFS
LIMIT_KNEE = 0.85        # soft limiter threshold: transients above this get tanh-compressed
GAIN_FLOOR_RMS = 0.01    # ~-40 dBFS: chunks below this are lead-in silence/breath; do not
                         # estimate utterance gain from them or it locks at 1.0 (too quiet)

# Sampling knobs for voice-quality experiments without code changes.
TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.1"))
REPETITION_PENALTY = float(os.environ.get("REPETITION_PENALTY", "1.1"))

audio_tokenizer: AudioTokenizerV2 | None = None
model: AutoModelForCausalLM | None = None
model_lock = asyncio.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global audio_tokenizer, model
    audio_tokenizer = AudioTokenizerV2(MODEL_ID, WAV_MODEL, WAV_CONFIG)
    # bf16 roughly doubles-to-quadruples token throughput vs the fp32 the
    # checkpoint loads as under "auto"; realtime factor was 1.13x in fp32,
    # which is what made playback choppy (underruns).
    dtype = torch.bfloat16 if torch.cuda.is_available() else "auto"
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=dtype)
    model = model.to(audio_tokenizer.device)
    print("[magana-tts] model loaded")
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/health")
async def health() -> dict:
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "version": os.environ.get("GIT_SHA", "unknown"),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }


def _soft_limit(audio: torch.Tensor) -> torch.Tensor:
    """Tanh-compress only the samples above LIMIT_KNEE so the utterance body can
    be lifted to TARGET_RMS while transient peaks land smoothly under 1.0."""
    magnitude = audio.abs()
    over = magnitude > LIMIT_KNEE
    if not bool(over.any()):
        return audio
    span = 1.0 - LIMIT_KNEE
    compressed = LIMIT_KNEE + span * torch.tanh((magnitude - LIMIT_KNEE) / span)
    return torch.where(over, torch.sign(audio) * compressed, audio)


def _resample_and_encode(audio: torch.Tensor, output_format: str, gain: float = 1.0) -> bytes:
    """Resample 24kHz float32 tensor (1, N) to the target format and return bytes."""
    if gain != 1.0:
        audio = _soft_limit(audio * gain)
    if output_format == "ulaw_8000":
        resampled = torchaudio.functional.resample(audio, 24000, 8000)
        pcm16 = (resampled.squeeze().cpu().numpy() * 32767).astype(np.int16)
        return audioop.lin2ulaw(pcm16.tobytes(), 2)
    # pcm_16000
    resampled = torchaudio.functional.resample(audio, 24000, 16000)
    pcm16 = (resampled.squeeze().cpu().numpy() * 32767).astype(np.int16)
    return pcm16.tobytes()


def _decode_chunk(
    codes: list[int],
    prev_overlap: list[int],
    is_first: bool,
) -> torch.Tensor:
    """
    Decode a batch of audio codes to a 24kHz float32 tensor (1, num_samples).
    Prepends prev_overlap to smooth boundary artifacts, then trims those
    samples so the caller receives only the intended chunk.
    """
    full_codes = prev_overlap + codes
    audio = audio_tokenizer.get_audio(full_codes)  # (1, num_samples) at 24kHz
    if not is_first and prev_overlap:
        trim = len(prev_overlap) * SAMPLES_PER_TOKEN
        audio = audio[:, trim:]
    return audio


def _agc_gain(prev: float | None, audio: torch.Tensor) -> float | None:
    """Slow AGC: per-chunk gain toward TARGET_RMS, smoothed to avoid pumping.
    A single utterance-level gain does not work for this model: measured
    dynamics show the first chunk can sit at -17 dBFS while the rest of the
    utterance trails off to -26..-33 dBFS, so a gain locked from the first
    energetic chunk leaves the tail inaudible. Silent chunks hold the previous
    gain. The soft limiter in _resample_and_encode absorbs transient peaks."""
    rms = float(audio.pow(2).mean().sqrt())
    if rms < GAIN_FLOOR_RMS:
        return prev
    target = max(1.0, min(TARGET_RMS / rms, 8.0))
    if prev is None:
        return target
    return 0.6 * prev + 0.4 * target


async def _synthesize(
    ws: WebSocket,
    text: str,
    voice_id: str,
    lang: str,
    output_format: str,
    clear_event: Event,
    loop: asyncio.AbstractEventLoop,
) -> None:
    """Run inference and stream PCM chunks to the WebSocket."""
    assert audio_tokenizer is not None and model is not None

    prompt = audio_tokenizer.create_prompt(text, lang=lang, speaker_name=voice_id)
    input_ids = audio_tokenizer.tokenize_prompt(prompt)

    streamer = TextIteratorStreamer(
        audio_tokenizer.tokenizer,
        skip_prompt=True,
        skip_special_tokens=False,
        timeout=30,
    )

    audio_queue: asyncio.Queue[bytes | Exception | None] = asyncio.Queue()

    def generate() -> None:
        try:
            model.generate(
                input_ids=input_ids,
                do_sample=True,
                temperature=TEMPERATURE,
                repetition_penalty=REPETITION_PENALTY,
                max_length=4000,
                streamer=streamer,
            )
        except Exception as exc:
            loop.call_soon_threadsafe(audio_queue.put_nowait, exc)

    def collect_and_decode() -> None:
        text_buf = ""
        codes_buf: list[int] = []
        prev_overlap: list[int] = []
        is_first = True
        gain: float | None = None

        def emit(codes: list[int]) -> None:
            nonlocal prev_overlap, is_first, gain
            audio = _decode_chunk(codes, prev_overlap, is_first)
            gain = _agc_gain(gain, audio)
            prev_overlap = codes[-OVERLAP_TOKENS:]
            is_first = False
            pcm = _resample_and_encode(audio, output_format, gain if gain is not None else 1.0)
            loop.call_soon_threadsafe(audio_queue.put_nowait, pcm)

        # Any exception here (including the streamer's 30s timeout raising
        # queue.Empty) MUST reach audio_queue: _synthesize awaits the queue
        # while holding model_lock, so a silently dead collector deadlocks
        # every subsequent synthesis on the pod.
        try:
            for token_text in streamer:
                if clear_event.is_set():
                    break
                text_buf += token_text
                # Extract complete <|N|> patterns; keep tail that may be partial
                matches = list(re.finditer(r"<\|(-?\d+)\|>", text_buf))
                if matches:
                    last_end = matches[-1].end()
                    text_buf = text_buf[last_end:]
                    codes_buf.extend(int(m.group(1)) for m in matches)

                # Small first chunk for fast first-audio, then larger chunks so the
                # decoder is called less often (throughput) with fewer boundaries.
                target = FIRST_CHUNK_TOKENS if is_first else CHUNK_TOKENS
                while len(codes_buf) >= target and not clear_event.is_set():
                    chunk = codes_buf[:target]
                    codes_buf = codes_buf[target:]
                    emit(chunk)
                    target = CHUNK_TOKENS
        except Exception as exc:
            loop.call_soon_threadsafe(audio_queue.put_nowait, exc)
            return

        # Flush remaining codes
        if codes_buf and not clear_event.is_set():
            try:
                emit(codes_buf)
            except Exception as exc:
                loop.call_soon_threadsafe(audio_queue.put_nowait, exc)
                return

        loop.call_soon_threadsafe(audio_queue.put_nowait, None)  # sentinel

    gen_thread = Thread(target=generate, daemon=True)
    collect_thread = Thread(target=collect_and_decode, daemon=True)
    gen_thread.start()
    collect_thread.start()

    while True:
        # Belt-and-braces: the streamer's own 30s timeout should surface any
        # stall as an Exception item, but if both worker threads die without
        # posting one, this must not hold model_lock forever.
        item = await asyncio.wait_for(audio_queue.get(), timeout=90)
        if item is None:
            break
        if isinstance(item, Exception):
            raise item
        await ws.send_bytes(item)


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket) -> None:
    await ws.accept()
    loop = asyncio.get_event_loop()
    clear_event = Event()

    try:
        raw = await ws.receive_text()
        handshake = json.loads(raw)
        voice_id: str = handshake.get("voiceId", "idera")
        output_format: str = handshake.get("outputFormat", "ulaw_8000")
        # Infer language from voice_id prefix; default to english
        lang = "english"
        for prefix in ("hausa", "igbo", "yoruba"):
            if voice_id.startswith(prefix):
                lang = prefix
                break

        if "text" in handshake:
            # Stream mode: single synthesis then done
            async with model_lock:
                clear_event.clear()
                await _synthesize(ws, handshake["text"], voice_id, lang, output_format, clear_event, loop)
            await ws.send_json({"type": "done"})
            return

        # Session mode: multiple text+flush cycles
        text_buf = ""
        async for raw_msg in ws.iter_text():
            msg = json.loads(raw_msg)
            msg_type = msg.get("type")

            if msg_type == "text":
                text_buf += msg.get("content", "")
            elif msg_type == "flush" and text_buf:
                text_to_speak = text_buf
                text_buf = ""
                clear_event.clear()
                async with model_lock:
                    await _synthesize(ws, text_to_speak, voice_id, lang, output_format, clear_event, loop)
                await ws.send_json({"type": "done"})
            elif msg_type == "clear":
                clear_event.set()
                text_buf = ""

    except WebSocketDisconnect:
        clear_event.set()
    except Exception as exc:
        clear_event.set()
        try:
            await ws.close(1011)
        except Exception:
            pass
        raise exc


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
