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
| `mood-warmup` | `curl` | — | One-shot gegen `/mood`. Der erste Aufruf kostet sonst ~15 s (numba-JIT von `librosa.pyin`) und läuft im Voice-Assistant in `DIARIZATION_TIMEOUT` — der erste Sprach-Turn nach einem Neustart verlöre seine Stimmungsanalyse. Muss über `voice-analysis` gehen, nicht direkt gegen `ser`: die Zeit steckt nicht im SER-Modell (Inferenz 25–28 ms) |
| `llamacpp-gemma` | `ghcr.io/ggml-org/llama.cpp:server-cuda` | 8090 | Nur `compose.gastonllm.yml`: kleines LLM für den Schalt-Aktuator des Voice-Assistant (Gemma, `-ngl 99`). Seit 2026-09-23 nur noch Rückweg, siehe nächste Zeile; seit 2026-09-28 gestoppt (`podman stop`), damit die 3060 Ti für `laya` frei ist |
| `llamacpp-gemma-vega` | `ghcr.io/ggml-org/llama.cpp:server-vulkan` | 8091 | Nur `compose.gastonllm.yml`: dasselbe Gemma auf der Vega-iGPU des Ryzen (Vulkan), damit die 3060 Ti frei wird. Braucht die udev-Regel `/etc/udev/rules.d/99-gastonllm-igpu.rules` (ACL auf `renderD128` für den rootless-User) — ohne sie rechnet llama.cpp still auf der CPU. Kaltstart 8–12 s je Prompt, aufgewärmt vom Voice-Assistant |
| `laya` | `./laya` (Build) | 8096 (nur `127.0.0.1`) | Nur `compose.gastonllm.yml`, seit 2026-09-29: Laya (Encoder + Entscheidungskopf) als Aktuator-Klassifikator im **Schattenbetrieb** des Voice-Assistant — urteilt mit, schaltet nie. Checkpoint ist auf die Ziele dieses Hauses trainiert und liegt unter `${LAYA_MODELLE}` auf dem Host, nicht im Image. ~1,4 GB VRAM. Das Image braucht `gcc`: torch 2.14 baut beim ersten Aufruf einen Triton-Kernel, ohne Compiler endet jede Anfrage mit HTTP 500. Nach einem Neubau `up -d --force-recreate laya`, sonst läuft der alte Container weiter |

## Projektname

`compose.gastonllm.yml` trägt `name: voice-gastonllm`. **Ein Projektname gehört
genau einer Compose-Datei.** Podman hängt ihn an Netz- und Volume-Namen; zwei
Stacks mit demselben Namen greifen sich gegenseitig die Container weg.

Bis 2026-08-01 hieß das Projekt hier `ai-stack-gastonllm` — der beim Repo-Split
mitgeschleppte Name des archivierten Vorgängers, den zeitweise *drei*
Verzeichnisse trugen. Ergebnis: vier Container liefen aus einem Handstart
heraus, ohne Compose-Netz, und der Voice-Assistant lief wochenlang unbemerkt
auf seinen Fallbacks. Details in `DECISIONS.md`.

Beim Umbenennen eines Projekts wandern die Volumes mit. Damit der Modell-Cache
das überlebt, sind sie explizit festgenagelt — deshalb steht im `volumes:`-Block
noch der alte Name, das ist Absicht:

```yaml
volumes:
  speaches-cache:
    name: ai-stack-gastonllm_speaches-cache
```

## Setup (neuer Host)

```bash
cp .env.example .env
# HF_CACHE-Pfad anpassen
```

> **`.env` nicht vergessen** — sie ist gitignored und kommt bei einem Klon oder
> Repo-Split *nicht* mit. Fehlt sie, ist `${HF_CACHE}` leer, der Volume-Mount
> von `ser` bricht, und Compose sieht „kaputt" aus. Genau das hat hier zum
> Handstart per `podman run` geführt — und der wiederum zu Containern ohne
> Compose-Netz. **Dienste immer über Compose starten, nie von Hand.**

```bash

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
