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
COPY entrypoint.sh .
COPY default_speakers/ ./default_speakers/
COPY default_speakers_local/ ./default_speakers_local/

RUN chmod +x /app/entrypoint.sh

ENV WAV_TOKENIZER_CONFIG=/app/wavtokenizer.yaml
ENV WAV_TOKENIZER_MODEL=/app/wavtokenizer.ckpt
ENV MODEL_ID=saheedniyi/YarnGPT2

EXPOSE 8000

CMD ["/app/entrypoint.sh"]
