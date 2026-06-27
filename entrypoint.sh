#!/bin/bash
set -e

YAML_URL="https://huggingface.co/novateur/WavTokenizer-medium-speech-75token/resolve/main/wavtokenizer_mediumdata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml"
CKPT_GDRIVE_ID="1-ASeEkrn4HY49yZWHTASgfGFNXdVnLTt"

if [ ! -f "$WAV_TOKENIZER_CONFIG" ]; then
    echo "[entrypoint] Downloading WavTokenizer config..."
    wget -q -O "$WAV_TOKENIZER_CONFIG" "$YAML_URL"
fi

if [ ! -f "$WAV_TOKENIZER_MODEL" ]; then
    echo "[entrypoint] Downloading WavTokenizer checkpoint (~1.5GB)..."
    pip install -q gdown
    gdown "$CKPT_GDRIVE_ID" -O "$WAV_TOKENIZER_MODEL"
fi

echo "[entrypoint] Starting server..."
exec python server.py
