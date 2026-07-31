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

## Handgestartete Container: warum sie entstehen und was sie anrichten (2026-08-01)

Vier von fünf Containern auf gastonllm liefen monatelang aus einem Handstart
per `podman run` heraus, nicht über Compose. Aufgefallen ist es erst, als
Diarization und Mood mit `[Errno 104] Connection reset by peer` ausstiegen.

**Warum der Handstart passierte** — und das ist der Teil, der leicht wieder
passiert: beim Repo-Split war die `.env` nicht mitgekommen. Sie ist gitignored,
`.env.example` allein reicht aber nicht. Ohne sie ist `${HF_CACHE}` leer, der
Volume-Mount von `ser` bricht, und Compose sieht schlicht kaputt aus. Der
Ausweg „dann eben von Hand starten" wirkt in dem Moment vernünftig.

**Was ein Handstart kostet.** Ohne `--network` landet der Container auf `pasta`
statt am Compose-Netz. Zwei Folgen, beide unsichtbar:

1. **Kein Container-DNS.** `voice-analysis` konnte `ser` nicht auflösen und
   fiel still auf seine Heuristik zurück — `/mood` antwortete weiter mit 200,
   nur ohne `ser`-Feld.
2. **`pasta` nimmt IPv6-Verbindungen an und setzt sie sofort zurück.** Auf
   diesem Host löst `localhost` zuerst nach `::1` auf. Alles, was `localhost`
   benutzte, war damit tot — der Voice-Assistant lief unbemerkt auf
   faster-whisper statt Speaches-GPU und auf Piper statt Speaches-TTS.

Das Tückische ist die Diagnose-Lage: `podman ps` meldet durchgehend
„Up (healthy)", und im Container-Log steht **nichts**, weil die Anfragen dort
nie ankommen. Wer so etwas sucht, prüft mit `curl -w "%{remote_ip}"` — steht da
`::1` und der Code ist `000`, ist es dieser Fall. `podman inspect <c> --format
'{{.HostConfig.NetworkMode}}'` zeigt `pasta` statt `bridge`, und
`{{.Config.CreateCommand}}` verrät den Handstart.

**Konsequenz:** Dienste ausschließlich über Compose starten. Nach jedem
Repo-Split oder Klon zuerst prüfen, ob die gitignorete `.env` da ist.

## Ein Projektname gehört genau einer Compose-Datei (2026-08-01)

Zeitweise trugen *drei* Verzeichnisse den Projektnamen `ai-stack-gastonllm`:
das archivierte `~/ai-stack`, ein toter Klon `~/voice-stack` und dieses Repo.
Podman hängt den Projektnamen an Netz- und Volume-Namen — solche Stacks nehmen
sich gegenseitig die Container weg, und ein Handstart aus dem falschen
Verzeichnis fällt nicht auf.

Seither: `name: voice-gastonllm` hier, `openclaw-gastonllm` für den Host-Stack
(infinity/Embeddings, `~/openclaw-gastonllm`), `fablab-llm` im Fablab-Repo.
Die Altverzeichnisse liegen unter `~/old-stack-files/` mit einem README, das
erklärt, warum man dort nichts starten soll.

**Umbenennen ist billig, wenn man die Volumes festnagelt.** Sonst leitet Podman
sie aus dem Projektnamen ab (`<projekt>_<volume>`) und man lädt den kompletten
Whisper-/Piper-Cache neu:

```yaml
volumes:
  speaches-cache:
    name: ai-stack-gastonllm_speaches-cache   # alter Name bleibt bewusst stehen
```

## Der erste `/mood`-Aufruf kostet 15 Sekunden — und zwar in librosa (2026-08-01)

Nach jedem Stack-Start braucht der erste `/mood`-Aufruf ~15 s, jeder weitere
0,2 s. Der Voice-Assistant wartet nur `DIARIZATION_TIMEOUT` und protokolliert
ein irreführendes „Mood: timed out" — der erste Sprach-Turn nach einem Neustart
verliert damit seine Stimmungsanalyse.

Die naheliegende Erklärung ist falsch: es ist **nicht** das SER-Modell und
**nicht** der CUDA-Warmlauf. Das wav2vec2-Modell liegt schon beim Import im
VRAM (`/health` meldet healthy) und die Inferenz braucht 25–28 ms. Die Zeit
steckt in `librosa.pyin` in `voice-analysis` — numba kompiliert die Funktion
beim ersten Aufruf.

Ein Warmup direkt gegen `ser:8002/ser` bringt deshalb nichts; genau das war der
erste, wirkungslose Versuch (gemessen: danach immer noch 15,02 s Timeout). Der
Warmup muss über `voice-analysis:8001/mood` gehen — der deckt beides ab, weil
`/mood` SER intern selbst aufruft. Danach: 1,30 s beim ersten Aufruf.
