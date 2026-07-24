# openclaw-voice-stack

Sprach-Pipeline für [OpenClaw](https://openclaw.ai) — Speech-to-Text,
Text-to-Speech, Diarization und Emotion Recognition, containerisiert über
Podman Compose. Läuft auf mehreren Hosts parallel, jeweils mit eigener
Compose-Datei.

> **Herkunft:** Herausgelöst am 2026-07-24 aus dem bisherigen, gemeinsamen
> `openclaw-local-ia-stack`-Repo (dort lief die Sprach-Pipeline zusammen mit
> LLM+Embeddings auf einem Host). LLM+Embeddings laufen jetzt separat, siehe
> [openclaw-fablab-llm](https://github.com/jochen/openclaw-fablab-llm). Volle
> Historie: [openclaw-local-ia-stack](https://github.com/jochen/openclaw-local-ia-stack)
> (Archiv, nicht mehr aktiv gepflegt).

## Aktuelle Hosts

| Host | Compose-Datei | GPU | Rolle |
|------|---------------|-----|-------|
| gastonllm | `compose.gastonllm.yml` | RTX 3060 Ti (Ampere) | Ersatz für rouven, wachsende Home-Hub-Rolle (perspektivisch eigenes kleines LLM) |
| rouven | `compose.rouven.yml` | GTX 1660 (Turing) | Ursprüngliche Test-Auslagerung, `float32`-STT-Workaround (keine Tensor Cores) |

## Dienste

| Dienst | Image | Port | Notes |
|---------|-------|------|-------|
| `speaches` | `ghcr.io/speaches-ai/speaches:0.9.0-rc.3-cuda` | 8000 | STT (faster-whisper/ctranslate2), TTS (Piper/ONNX), Diarization; Compute-Type abhängig von GPU-Architektur, siehe `DECISIONS.md` |
| `speaches-warmup` | `alpine`/`curl` | — | One-shot: lädt STT/TTS/Diarization-Modelle beim Start, beendet sich danach |
| `ser` | `./ser` (Build) | 8002 | Speech Emotion Recognition, `POST /ser` (WAV → arousal/valence/dominance + label), Modell `audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim` (~600 MiB VRAM) |
| `voice-analysis` | `./voice-analysis` (Build) | 8001 | CPU-only; Re-STT + Texttreue (WER/CER) + Timing + Prosodie; `/mood` SER-backed (ruft `ser:8002` intern) |

## Setup (neuer Host)

```bash
cp .env.example .env
# HF_CACHE-Pfad anpassen

# Passende Compose-Datei fürs jeweilige Modell/GPU als Vorlage nehmen,
# insb. WHISPER__COMPUTE_TYPE pruefen (siehe DECISIONS.md):
cp compose.gastonllm.yml compose.<neuer-host>.yml   # oder compose.rouven.yml als Basis

podman-compose -f compose.<neuer-host>.yml up -d
```

Persistente Volumes (überleben Neustarts):
- **`speaches-cache`** — HuggingFace-Weights
- **`speaches-cuda-cache`** — kompilierte CUDA-Kernels (verhindert JIT-Delay
  beim ersten Aufruf, besonders relevant auf Blackwell-GPUs)

## voice-analysis — `/analyze`-Contract

`POST http://<host>:8001/analyze` (multipart/form-data)

| Feld | Typ | Pflicht | Beschreibung |
|------|-----|---------|--------------|
| `file` | WAV | ja | PCM16 mono, 16 kHz oder 22 kHz |
| `intended` | string | ja | Erwarteter Sprechtext |
| `language` | string | nein | ISO-639-1, default `de` |

Antwort-Felder:
- **`text_fidelity`** — `wer`, `cer`, `match` (WER < 0.15 gilt als Treffer)
- **`timing`** — `duration_s`, `word_count`, `words_per_sec`, `pauses`, `pause_total_s`
- **`prosody`** — `f0_mean/std/min/max_hz`, `rms_mean/std`
- **`mood_proxy`** — `label` aus {neutral, aufgeregt/genervt, müde/traurig}, `hint`

## voice-analysis — `/mood`-Contract

`POST http://<host>:8001/mood` (multipart/form-data) — schnelle Stimmungsanalyse
von Nutzer-Audio, kein STT/Textvergleich.

| Feld | Typ | Pflicht | Beschreibung |
|------|-----|---------|--------------|
| `file` | WAV | ja | PCM16 mono, beliebige Sample-Rate |

Antwort-Felder:
- **`prosody`** — wie oben
- **`mood_proxy`** — `label` aus {neutral, genervt/verärgert, aufgeregt/freudig, müde/traurig}, `hint`
- **`ser`** — Rohwerte vom SER-Dienst: `arousal`, `valence`, `dominance` (0–1), `label`, `infer_ms`, `device`; `null` wenn Dienst nicht erreichbar

Label-Mapping (Schwellen in `ser/app.py` justierbar):

| Bedingung | Label |
|-----------|-------|
| arousal > 0.60 und valence < 0.45 | `genervt/verärgert` |
| arousal > 0.60 und valence ≥ 0.55 | `aufgeregt/freudig` |
| arousal < 0.40 und valence < 0.45 | `müde/traurig` |
| sonst | `neutral` |

## wakeword-studio

GPU-gestützte Trainings-/Evaluations-Tools für eigene Wakeword-Modelle
(openWakeWord-kompatibel). Details: `wakeword-studio/README.md`. Genutzt vom
Client-seitigen [openclaw_voice_assist](https://github.com/jochen/openclaw_voice_assist)
(`python -m wakeword_studio record/score`).

## Stack-Verwaltung

```bash
podman-compose -f compose.<host>.yml up -d
podman-compose -f compose.<host>.yml down
podman ps
podman logs -f speaches
podman exec speaches nvidia-smi   # VRAM (Host-nvidia-smi ggf. kaputt)
```

## Verwandte Repositories

- [openclaw-fablab-llm](https://github.com/jochen/openclaw-fablab-llm) — LLM + Embeddings
- [openclaw_voice_assist](https://github.com/jochen/openclaw_voice_assist) — Voice-Pipeline-Client (Wakeword/STT/TTS-Orchestrierung, läuft auf dem OpenClaw-Host selbst)
