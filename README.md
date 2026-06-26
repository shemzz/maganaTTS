# Magana TTS

Magana TTS is a Nigerian-accented text-to-speech service powering the Magana
voice AI platform. It is forked from [YarnGPT](https://github.com/saheedniyi02/yarngpt)
and extended with a streaming WebSocket inference server for real-time voice calls.

## What's Inside

- `audiotokenizer.py` — core tokenizer classes (upstream from YarnGPT)
- `server.py` — FastAPI + WebSocket streaming inference server
- `Dockerfile` — RunPod GPU deployment image
- `default_speakers/` — English speaker reference audio codes
- `default_speakers_local/` — Hausa, Igbo, Yoruba speaker codes

## Running Locally (for development)

```bash
pip install -r requirements-server.txt
python server.py
```

The server starts on `ws://localhost:8000/ws`.

## Deploying to RunPod

1. Build the Docker image and push to a registry (Docker Hub or GHCR).
2. Create a RunPod GPU pod (T4 recommended) using the image.
3. Set `MAGANA_TTS_URL=wss://<pod-id>-8000.proxy.runpod.net/ws` in the Magana API `.env`.

## Supported Voices

Magana TTS exposes 12 curated personas. The underlying speaker IDs are
internal implementation details managed by the Magana API.
