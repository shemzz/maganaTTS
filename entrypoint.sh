#!/bin/bash
set -e

# YarnGPT2 emits codes in the WavTokenizer *large-speech* codebook, paired with the
# medium-speech-75token config (see python-wrapper/yarngpt/core.py). Decoding with any
# other WavTokenizer variant (e.g. the small-data one) produces speech-like gibberish.
YAML_URL="https://huggingface.co/novateur/WavTokenizer-medium-speech-75token/resolve/main/wavtokenizer_mediumdata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml"
# Official YarnGPT source for the ckpt is Google Drive (the HF repo now only hosts an
# incompatible v2). Override with WAV_TOKENIZER_MODEL_URL to serve it from own storage.
CKPT_GDRIVE_ID="1-ASeEkrn4HY49yZWHTASgfGFNXdVnLTt"
CKPT_MIN_BYTES=1700000000  # real ckpt is ~1.75GB; a Drive quota/error page is a few KB

if [ ! -f "$WAV_TOKENIZER_CONFIG" ]; then
    echo "[entrypoint] Downloading WavTokenizer config..."
    curl -L --fail --retry 5 --retry-delay 5 -o "$WAV_TOKENIZER_CONFIG" "$YAML_URL"
else
    echo "[entrypoint] Found WavTokenizer config at $WAV_TOKENIZER_CONFIG"
fi

download_ckpt() {
    if [ -n "$WAV_TOKENIZER_MODEL_URL" ]; then
        echo "[entrypoint] Downloading WavTokenizer checkpoint from WAV_TOKENIZER_MODEL_URL..."
        curl -L --fail --retry 5 --retry-delay 5 -o "$WAV_TOKENIZER_MODEL" "$WAV_TOKENIZER_MODEL_URL"
        return
    fi
    echo "[entrypoint] Downloading WavTokenizer checkpoint from Google Drive (~1.75GB)..."
    # Large Drive files sit behind a confirm interstitial; extract the uuid then download.
    local cookie_jar interstitial uuid
    cookie_jar=$(mktemp)
    interstitial=$(curl -sL -c "$cookie_jar" "https://drive.google.com/uc?export=download&id=${CKPT_GDRIVE_ID}")
    uuid=$(echo "$interstitial" | grep -oE 'name="uuid" value="[^"]*"' | cut -d'"' -f4)
    curl -L --fail --retry 10 --retry-delay 10 -C - -b "$cookie_jar" \
        -o "$WAV_TOKENIZER_MODEL" \
        "https://drive.usercontent.google.com/download?id=${CKPT_GDRIVE_ID}&export=download&confirm=t&uuid=${uuid}"
    rm -f "$cookie_jar"
}

ckpt_size() { stat -c%s "$WAV_TOKENIZER_MODEL" 2>/dev/null || stat -f%z "$WAV_TOKENIZER_MODEL" 2>/dev/null || echo 0; }

if [ ! -f "$WAV_TOKENIZER_MODEL" ] || [ "$(ckpt_size)" -lt "$CKPT_MIN_BYTES" ]; then
    rm -f "$WAV_TOKENIZER_MODEL"
    download_ckpt
    if [ "$(ckpt_size)" -lt "$CKPT_MIN_BYTES" ]; then
        echo "[entrypoint] ERROR: checkpoint download came back $(ckpt_size) bytes (< $CKPT_MIN_BYTES). Refusing to start with a bad codec." >&2
        exit 1
    fi
else
    echo "[entrypoint] Found WavTokenizer checkpoint at $WAV_TOKENIZER_MODEL"
fi

echo "[entrypoint] Starting server..."
exec python -u server.py
