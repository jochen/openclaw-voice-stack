FROM python:3.11-slim

WORKDIR /app

# System-Abhängigkeiten für Audio (soundfile/librosa)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    libgomp1 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# torch separat installieren. Standard cu128 (Turing/Ampere: rouven, gastonllm).
# misterhandy (Tesla M60, sm_52) setzt per build-args cu118 + 2.6.0: torch fuer
# cu12x enthaelt keine sm_52-Kernel mehr, das Modell liefe still auf der CPU.
ARG TORCH_VERSION=2.7.0
ARG TORCH_INDEX=cu128
RUN pip install --no-cache-dir torch==${TORCH_VERSION} --index-url https://download.pytorch.org/whl/${TORCH_INDEX}

# Rest
RUN pip install --no-cache-dir \
    transformers==4.51.3 \
    accelerate>=0.26.0 \
    soundfile>=0.12.1 \
    librosa>=0.10.0 \
    "numpy>=1.24,<2.0" \
    "fastapi>=0.111.0" \
    "uvicorn[standard]>=0.29.0" \
    "python-multipart>=0.0.9" \
    httpx>=0.27.0

COPY app.py .

EXPOSE 8002
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8002"]
