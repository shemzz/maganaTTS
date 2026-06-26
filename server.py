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

CHUNK_TOKENS = 25
OVERLAP_TOKENS = 2
SAMPLES_PER_TOKEN = 320  # 24000 Hz / 75 tokens per second

audio_tokenizer: AudioTokenizerV2 | None = None
model: AutoModelForCausalLM | None = None
model_lock = asyncio.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global audio_tokenizer, model
    audio_tokenizer = AudioTokenizerV2(MODEL_ID, WAV_MODEL, WAV_CONFIG)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype="auto")
    model = model.to(audio_tokenizer.device)
    print("[magana-tts] model loaded")
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "model_loaded": model is not None}


def _resample_and_encode(audio: torch.Tensor, output_format: str) -> bytes:
    """Resample 24kHz float32 tensor (1, N) to the target format and return bytes."""
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
    output_format: str,
) -> bytes:
    """
    Decode a batch of audio codes to PCM bytes.
    Prepends prev_overlap to smooth boundary artifacts, then trims those
    samples before encoding so the caller receives only the intended chunk.
    """
    full_codes = prev_overlap + codes
    audio = audio_tokenizer.get_audio(full_codes)  # (1, num_samples) at 24kHz
    if not is_first and prev_overlap:
        trim = len(prev_overlap) * SAMPLES_PER_TOKEN
        audio = audio[:, trim:]
    return _resample_and_encode(audio, output_format)


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
                temperature=0.1,
                repetition_penalty=1.1,
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

            while len(codes_buf) >= CHUNK_TOKENS and not clear_event.is_set():
                chunk = codes_buf[:CHUNK_TOKENS]
                codes_buf = codes_buf[CHUNK_TOKENS:]
                try:
                    pcm = _decode_chunk(chunk, prev_overlap, is_first, output_format)
                    prev_overlap = chunk[-OVERLAP_TOKENS:]
                    is_first = False
                    loop.call_soon_threadsafe(audio_queue.put_nowait, pcm)
                except Exception as exc:
                    loop.call_soon_threadsafe(audio_queue.put_nowait, exc)
                    return

        # Flush remaining codes
        if codes_buf and not clear_event.is_set():
            try:
                pcm = _decode_chunk(codes_buf, prev_overlap, is_first, output_format)
                loop.call_soon_threadsafe(audio_queue.put_nowait, pcm)
            except Exception as exc:
                loop.call_soon_threadsafe(audio_queue.put_nowait, exc)
                return

        loop.call_soon_threadsafe(audio_queue.put_nowait, None)  # sentinel

    gen_thread = Thread(target=generate, daemon=True)
    collect_thread = Thread(target=collect_and_decode, daemon=True)
    gen_thread.start()
    collect_thread.start()

    while True:
        item = await audio_queue.get()
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
