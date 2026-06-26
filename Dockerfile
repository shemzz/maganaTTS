FROM --platform=linux/amd64 pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# Install system deps (build-essential needed for pesq C extension)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget curl build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements-server.txt .
RUN pip install --no-cache-dir -r requirements-server.txt

# Copy source
COPY audiotokenizer.py .
COPY server.py .
COPY default_speakers/ ./default_speakers/
COPY default_speakers_local/ ./default_speakers_local/

# Download WavTokenizer artifacts at build time
RUN mkdir -p /app/wav_models && \
    wget -q -O /app/wavtokenizer.yaml \
    "https://huggingface.co/novateur/WavTokenizer-medium-speech-75token/resolve/main/wavtokenizer_mediumdata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml" && \
    pip install gdown && \
    gdown 1-ASeEkrn4HY49yZWHTASgfGFNXdVnLTt -O /app/wavtokenizer.ckpt

ENV WAV_TOKENIZER_CONFIG=/app/wavtokenizer.yaml
ENV WAV_TOKENIZER_MODEL=/app/wavtokenizer.ckpt
ENV MODEL_ID=saheedniyi/YarnGPT2

EXPOSE 8000

CMD ["python", "server.py"]
