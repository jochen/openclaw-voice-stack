# Architecture Decisions & Lessons Learned

Entscheidungen und Stolperfallen speziell zur Sprach-Pipeline (speaches/ser/
voice-analysis). Für die LLM/Embeddings-Entscheidungen (ollama/vLLM/lms-Wahl
etc.) siehe [openclaw-fablab-llm/DECISIONS.md](https://github.com/jochen/openclaw-fablab-llm/blob/main/DECISIONS.md).

---

## int8 vs. float32 für Whisper-STT — abhängig von der GPU-Architektur

`WHISPER__COMPUTE_TYPE` in speaches ist **nicht** universell auf `int8`
setzbar — das hängt an Tensor Cores:

- **Ampere und neuer** (z. B. RTX 3060 Ti, RTX 5060 Ti): `int8` funktioniert
  korrekt und schnell.
- **Turing ohne Tensor Cores** (z. B. GTX 1660, TU116): `int8` liefert
  **stillschweigend falschen Text** (Whisper-Halluzination, kein Crash/Fehler!
  z. B. immer "Das war's für heute. Bis zum nächsten Mal." unabhängig vom
  echten Audioinhalt). `float16` ist noch schlimmer (Garbage-Tokens). Einzige
  verifiziert korrekte Option: `WHISPER__COMPUTE_TYPE=float32` (langsamer,
  RTF ~0.78 statt ~0.13, aber noch schneller als Echtzeit).

**Vorgehen bei neuem Host:** GPU-Architektur (Tensor Cores ja/nein) prüfen,
bevor man `int8` übernimmt. Im Zweifel: STT-Testtranskription mit bekanntem
Audioinhalt gegen `WHISPER__INFERENCE_DEVICE=cpu` vergleichen — weicht das
Ergebnis ab, ist es ein GPU-spezifisches Problem, kein Audio-/Modellproblem.

## Diarization NaN-Crash-Patch

`/v1/audio/diarization` kann mit "matrix contains invalid numeric entries"
crashen, wenn ein Cluster ohne gültige Trainings-Embeddings einen NaN-Centroid
erzeugt (`onnx_diarization/pipeline.py`, `linear_sum_assignment`). Kein
Upstream-Fix im aktuellen `0.9.0-rc.3`-Tag verfügbar (der main-Branch hat den
kompletten Diarization-Unterbau auf `pyannote` umgestellt, aber das bringt
neue Blocker: `pyannote/speaker-diarization-community-1` ist HF-gated, plus
Breaking-Change im Request-Schema). Minimaler Patch stattdessen:
`patches/onnx_diarization_pipeline.py` wird per Volume-Mount über die
Original-Datei im Container gelegt (siehe `compose.*.yml`), ergänzt eine
Guard-Zeile (`np.nan_to_num` vor `linear_sum_assignment`, gleiches Muster wie
in `clustering/vbx.py::constrained_argmax` im selben Package).

## GPU-Zuweisung für speaches

`scripts/speaches-entrypoint.sh`: feste Zuweisung via `SPEACHES_GPU` (Compose-
Env) hat Vorrang vor dynamischer "meiste freie VRAM"-Wahl. Die dynamische Wahl
war fehleranfällig — direkt nach Stack-Start gewinnt immer die GPU, die später
vom LLM vollgepackt wird, was zu Whisper-OOM/500ern führte, sobald ein großer
Kontext geladen war. Bei Single-GPU-Hosts (rouven, gastonllm) ist das
irrelevant, `SPEACHES_GPU` einfach weglassen.

## Warum kein gemeinsames `compose.yml`

Jeder Host hat ein eigenes `compose.<hostname>.yml`, weil die Details pro
Host divergieren (GPU-Architektur → Compute-Type, Modellauswahl je nach
VRAM, ob ein eigenes kleines LLM mitläuft). Ein generisches `compose.yml`
würde suggerieren, es passe überall — tut es nicht.
