FROM --platform=linux/amd64 pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime

WORKDIR /app

ENV PYTHONUNBUFFERED=1

# Install system deps (build-essential needed for pesq C extension)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget curl build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements-server.txt .
RUN grep -vE "^(torch|torchaudio)(\[.*\])?(==.*)?$" requirements-server.txt > requirements-server-docker.txt \
    && pip install --no-cache-dir -r requirements-server-docker.txt

# Copy source
COPY audiotokenizer.py .
COPY server.py .
COPY entrypoint.sh .
COPY default_speakers/ ./default_speakers/
COPY default_speakers_local/ ./default_speakers_local/

RUN chmod +x /app/entrypoint.sh && mkdir -p /app/models

ENV WAV_TOKENIZER_CONFIG=/app/models/wavtokenizer_smalldata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml
ENV WAV_TOKENIZER_MODEL=/app/models/WavTokenizer_small_320_24k_4096.ckpt
ENV MODEL_ID=saheedniyi/YarnGPT2

EXPOSE 8000

CMD ["/app/entrypoint.sh"]
