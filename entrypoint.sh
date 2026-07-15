#!/bin/bash
set -e

YAML_URL="https://huggingface.co/novateur/WavTokenizer/resolve/main/wavtokenizer_smalldata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml"
CKPT_URL="https://huggingface.co/novateur/WavTokenizer/resolve/main/WavTokenizer_small_320_24k_4096.ckpt"

if [ ! -f "$WAV_TOKENIZER_CONFIG" ]; then
    echo "[entrypoint] Downloading WavTokenizer config..."
    curl -L --fail --retry 5 --retry-delay 5 -o "$WAV_TOKENIZER_CONFIG" "$YAML_URL"
else
    echo "[entrypoint] Found WavTokenizer config at $WAV_TOKENIZER_CONFIG"
fi

if [ ! -f "$WAV_TOKENIZER_MODEL" ]; then
    echo "[entrypoint] Downloading WavTokenizer checkpoint (~1.5GB)..."
    curl -L --fail --retry 5 --retry-delay 5 -o "$WAV_TOKENIZER_MODEL" "$CKPT_URL"
else
    echo "[entrypoint] Found WavTokenizer checkpoint at $WAV_TOKENIZER_MODEL"
fi

echo "[entrypoint] Starting server..."
exec python -u server.py
