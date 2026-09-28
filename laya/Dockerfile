FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Dieselbe torch-Fassung wie beim Training (tools/laya_aktuator_train.py im
# Repo openclaw_voice_assist). cu128 gibt es nur bis torch 2.11.
RUN pip install --no-cache-dir "torch==2.14.0+cu130" \
    --index-url https://download.pytorch.org/whl/cu130
RUN pip install --no-cache-dir "laya[serve]==0.3.21"

# torch 2.14 leitet einzelne Operationen (bmm in der RoPE von ModernBERT) auf
# Triton-Kernel um, und Triton baut beim ersten Aufruf ein Hilfsmodul mit dem
# C-Compiler. Ohne gcc: jede Anfrage HTTP 500 "Failed to find C compiler"
# (2026-09-29). Hinter den pip-Schichten, damit ein Umbau torch nicht neu laedt.
RUN apt-get update && apt-get install -y --no-install-recommends gcc libc6-dev \
    && rm -rf /var/lib/apt/lists/*

COPY serve.py .

EXPOSE 8096
CMD ["python", "serve.py"]
