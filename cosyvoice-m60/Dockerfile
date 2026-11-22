# CosyVoice3-Dienst fuer Tesla M60 (sm_52, misterhandy): reines PyTorch fp32, ohne vLLM (braucht sm_70+)
# und ohne TensorRT (braucht sm_75+). Messungen und Begruendungen: gitlab fablab/misterhandy-host,
# versuche/cosyvoice-m60/ERGEBNIS.md. CosyVoice-Stand wie auf dem LLM-Rechner (.126): 074ca6d.
FROM python:3.10-slim
RUN apt-get update && apt-get install -y --no-install-recommends git ffmpeg libsndfile1 sox build-essential curl \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118 --extra-index-url https://pypi.org/simple \
    && printf "torch==2.6.0+cu118\ntorchaudio==2.6.0+cu118\n" > /etc/pip-constraints.txt
# Jede weitere pip-Installation darf torch nicht anfassen (sonst cu12/13 ohne sm_52)
ENV PIP_CONSTRAINT=/etc/pip-constraints.txt
RUN git clone https://github.com/FunAudioLLM/CosyVoice.git /opt/CosyVoice \
    && cd /opt/CosyVoice && git checkout 074ca6d && git submodule update --init --recursive
# openai-whisper 20231117 gibt es nur als sdist, setup.py braucht pkg_resources (fehlt in neuen setuptools)
RUN pip install --no-cache-dir "setuptools<70" wheel && pip install --no-cache-dir --no-build-isolation --no-deps openai-whisper==20231117 && pip install --no-cache-dir numba tiktoken more-itertools tqdm
RUN cd /opt/CosyVoice && grep -v -E "^(--extra-index-url|openai-whisper|deepspeed|tensorrt|onnxruntime|torch|gradio|grpcio|tensorboard|fastapi-cli)" requirements.txt > /tmp/req.txt \
    && pip install --no-cache-dir -r /tmp/req.txt onnxruntime==1.18.0 huggingface_hub
ENV PYTHONPATH=/opt/CosyVoice:/opt/CosyVoice/third_party/Matcha-TTS
WORKDIR /opt/CosyVoice
# Kontrolle: torch muss 2.6.0+cu118 bleiben (sonst still keine sm_52-Kernel)
RUN python -c "import torch, torchaudio; assert torch.__version__ == '2.6.0+cu118', torch.__version__; assert torchaudio.__version__.startswith('2.6.0'), torchaudio.__version__; print('torch ok', torch.__version__)"
COPY dienst.py /opt/dienst.py
EXPOSE 8010
CMD ["python", "-u", "/opt/dienst.py"]
