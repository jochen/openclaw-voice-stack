"""CosyVoice3 als Speaches-kompatibler TTS-Dienst für 2x Tesla M60 (misterhandy), streamend.

Gleiche HTTP-Schnittstelle wie cosy_dienst.py auf dem LLM-Rechner (.126), damit jeder
Voice-Assistant (Misterhandy im FabLab, später Gaston zu Hause) nur die URL wechseln muss:

POST /v1/audio/speech  {"input", "voice", "model"?, "response_format": "wav",
                        "sample_rate"?: 24000|48000, "speed"?: 0.5–2.0}
  speed != 1: CosyVoice kann Tempo nur ohne Streaming (staucht das Spektrogramm vor
  dem Vocoder, Tonhöhe bleibt). Dann kommt der Satz am Stück statt in Schüben; der
  Satz-Planer im Assistenten behandelt das wie einen einzigen großen Schub.
  -> WAV (16 bit mono) mit Platzhalter-Längen, Audio fließt, sobald es entsteht.
GET  /health, /v1/models, /v1/voices
POST /v1/voices        {"id": "<namensraum>/<name>-<pruefwert>", "text", "wav_b64"}
  -> Stimme zur Laufzeit anlegen (nur im Speicher). Die Assistenten verwalten ihre Stimmen
     selbst und laden sie hoch, wenn eine Anfrage 404 bekommt. Der Namensraum trennt die
     Assistenten, mehrere können den Dienst gleichzeitig nutzen.

Anders als auf .126 (vLLM + TensorRT gibt es für Maxwell/sm_52 nicht) — gemessen 2026-11-22,
gitlab fablab/misterhandy-host versuche/cosyvoice-m60/ERGEBNIS.md:
- reines PyTorch fp32 (torch 2.6+cu118), LLM auf LLM_GPU, Flow + HiFi-GAN auf FLOW_GPU,
- Flow mit SCHRITTE Diffusionsschritten (5 statt 10),
- LLM-Decode als CUDA-Graph mit statischem KV-Cache (tokengleich zu eager geprüft),
- **Pipeline:** Anfragen laufen gleichzeitig; nur das LLM ist gesperrt, und zwar FIFO in der
  Reihenfolge, in der die Anfragen ihren LLM-Teil beginnen. Schickt ein Assistent den nächsten
  Satz, während der vorige noch spricht, erzeugt das LLM die Sätze am Stück, der Flow rechnet
  parallel auf der anderen GPU. So ist der Durchsatz schneller als Echtzeit (RTF ~0,96),
  einzelne Sätze allein sind es nicht (RTF ~1,2) -> der Assistent plant Pausen zwischen Sätzen.
"""
import base64
import collections
import http.server
import json
import os
import re
import struct
import sys
import tempfile
import threading
import time

import numpy as np
import soundfile as sf
import torch
import torchaudio
from scipy.signal import firwin, lfilter
from transformers import StaticCache

COSY_DIR = os.environ.get("COSY_DIR", "/opt/CosyVoice")
STIMMEN_DIR = os.environ.get("STIMMEN_DIR", "/stimmen")
MODELL_DIR = os.environ.get("COSY_MODELL", "/modelle/Fun-CosyVoice3-0.5B-2512")
PORT = int(os.environ.get("PORT", "8010"))
LLM_GPU = torch.device(f"cuda:{os.environ.get('LLM_GPU', '1')}")
FLOW_GPU = torch.device(f"cuda:{os.environ.get('FLOW_GPU', '0')}")
SCHRITTE = int(os.environ.get("SCHRITTE", "5"))
GRAPH = os.environ.get("GRAPH", "1") == "1"
MAXLEN = int(os.environ.get("MAXLEN", "2048"))   # KV-Cache-Länge (Prompt + Sprach-Token eines Satzes)
HOP = 12
MODELL_NAME = "cosyvoice3"
PROMPT = "You are a helpful assistant.<|endofprompt|>"

os.chdir(COSY_DIR)
sys.path.insert(0, COSY_DIR)
sys.path.insert(0, os.path.join(COSY_DIR, "third_party/Matcha-TTS"))


def _load(p, *a, **k):
    d, sr = sf.read(p, dtype="float32", always_2d=True)
    return torch.from_numpy(d.T.copy()), sr


torchaudio.load = _load

from cosyvoice.cli.cosyvoice import AutoModel  # noqa: E402
import cosyvoice.cli.model as _cm  # noqa: E402


class _Zeit:
    """Abfrage-Schlaf 10 ms statt 100 ms (wie auf .126)."""

    def __getattr__(self, n):
        return getattr(time, n)

    def sleep(self, s):
        time.sleep(min(s, 0.01))


_cm.time = _Zeit()

t0 = time.time()
torch.cuda.set_device(FLOW_GPU)          # Frontend-Merkmale + Stimmen landen auf dem Flow-Gerät
M = AutoModel(model_dir=MODELL_DIR, load_vllm=False, load_trt=False, fp16=False)
SR = M.sample_rate
m = M.model
lm = m.llm
qwen = lm.llm.model.model                # HF Qwen2Model, last_hidden_state = normierter letzter Zustand

# --- Aufteilung auf die GPUs
lm.to(LLM_GPU)
m.flow.to(FLOW_GPU)
m.hift.to(FLOW_GPU)
m.device = FLOW_GPU
_llm_inference = lm.inference


def _llm_inference_geraet(*a, **k):
    a = [x.to(LLM_GPU) if torch.is_tensor(x) else x for x in a]
    k = {n: (v.to(LLM_GPU) if torch.is_tensor(v) else v) for n, v in k.items()}
    return _llm_inference(*a, **k)


lm.inference = _llm_inference_geraet

# --- Diffusionsschritte des Flows
_dec_forward = m.flow.decoder.forward


def _dec_schritte(*a, **k):
    if "n_timesteps" in k:
        k["n_timesteps"] = SCHRITTE
    elif len(a) >= 3:
        a = list(a)
        a[2] = SCHRITTE
    return _dec_forward(*a, **k)


m.flow.decoder.forward = _dec_schritte


class FifoSperre:
    """Sperre, die Wartende in Ankunftsreihenfolge bedient (threading.Lock garantiert das nicht)."""

    def __init__(self):
        self._cv = threading.Condition()
        self._naechste = 0
        self._dran = 0

    def __enter__(self):
        with self._cv:
            nr = self._naechste
            self._naechste += 1
            while nr != self._dran:
                self._cv.wait()

    def __exit__(self, *a):
        with self._cv:
            self._dran += 1
            self._cv.notify_all()


LLM_SPERRE = FifoSperre()
_orig_wrapper = lm.inference_wrapper


class GraphDecode:
    """Ein Decode-Schritt (Qwen2 + llm_decoder + log_softmax) als CUDA-Graph mit statischem KV-Cache.

    Ein Objekt für alle Anfragen: der Cache wird je Satz geleert (reset() behält die Adressen),
    die LLM-Sperre sorgt dafür, dass immer nur ein Satz ihn benutzt.
    """

    def __init__(self):
        self.cache = StaticCache(config=qwen.config, max_batch_size=1, max_cache_len=MAXLEN,
                                 device=LLM_GPU, dtype=torch.float32)
        self.emb = torch.zeros(1, 1, qwen.config.hidden_size, device=LLM_GPU)
        self.pos = torch.zeros(1, dtype=torch.long, device=LLM_GPU)
        self.graph = None
        self.out = None

    def _schritt(self):
        h = qwen(inputs_embeds=self.emb, past_key_values=self.cache, cache_position=self.pos,
                 use_cache=True).last_hidden_state
        return lm.llm_decoder(h[:, -1]).log_softmax(dim=-1)

    def _aufnehmen(self, start):
        self.pos.fill_(start)
        s = torch.cuda.Stream(device=LLM_GPU)
        s.wait_stream(torch.cuda.current_stream(LLM_GPU))
        with torch.cuda.stream(s):
            for _ in range(3):
                self._schritt()
        torch.cuda.current_stream(LLM_GPU).wait_stream(s)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.out = self._schritt()

    @torch.inference_mode()
    def erzeugen(self, lm_input, sampling, min_len, max_len, uuid):
        with torch.cuda.device(LLM_GPU):
            self.cache.reset()
            p = lm_input.shape[1]
            max_len = min(max_len, MAXLEN - p - 1)
            h = qwen(inputs_embeds=lm_input, past_key_values=self.cache,
                     cache_position=torch.arange(p, device=LLM_GPU), use_cache=True).last_hidden_state
            logp = lm.llm_decoder(h[:, -1]).log_softmax(dim=-1)
            if self.graph is None:
                self._aufnehmen(p)        # schreibt Müll an Position p, der erste Schritt überschreibt ihn
            out_tokens = []
            for i in range(max_len):
                top = lm.sampling_ids(logp.squeeze(dim=0), out_tokens, sampling,
                                      ignore_eos=True if i < min_len else False)
                if top in lm.stop_token_ids:
                    break
                yield top
                out_tokens.append(top)
                self.emb.copy_(lm.speech_embedding.weight[top].reshape(1, 1, -1))
                self.pos.fill_(p + i)
                self.graph.replay()
                logp = self.out


GD = GraphDecode() if GRAPH else None


def _llm_wrapper(*a, **k):
    with LLM_SPERRE:
        if GD is not None:
            yield from GD.erzeugen(*a, **k)
        else:
            yield from _orig_wrapper(*a, **k)


lm.inference_wrapper = _llm_wrapper

STIMMEN_SPERRE = threading.Lock()


def stimme_laden(text, wav, kennung):
    with STIMMEN_SPERRE:
        M.add_zero_shot_spk(PROMPT + text, wav, kennung)
        for key, v in list(M.frontend.spk2info[kennung].items()):
            if torch.is_tensor(v):
                M.frontend.spk2info[kennung][key] = v.to(FLOW_GPU)


STIMMEN = []
for datei in sorted(os.listdir(STIMMEN_DIR)):
    if datei.endswith(".wav") and os.path.exists(os.path.join(STIMMEN_DIR, datei[:-4] + ".txt")):
        name = datei[:-4]
        stimme_laden(open(os.path.join(STIMMEN_DIR, name + ".txt")).read().strip(),
                     os.path.join(STIMMEN_DIR, datei), name)
        STIMMEN.append(name)
if not STIMMEN:
    sys.exit(f"keine Stimmen in {STIMMEN_DIR}")

for _ in range(2):                       # Graph aufnehmen + CUDA-JIT, bevor Anfragen kommen
    m.token_hop_len = HOP
    for _o in M.inference_zero_shot("Hallo, das ist ein Test.", "", "", zero_shot_spk_id=STIMMEN[0], stream=True):
        pass
print(f"bereit nach {time.time() - t0:.0f} s, Stimmen {STIMMEN}, {SR} Hz, Port {PORT}, Modell {MODELL_DIR}, "
      f"LLM {LLM_GPU}, Flow {FLOW_GPU}, {SCHRITTE} Schritte, Graph {'an' if GD else 'aus'}", flush=True)

KENNUNG = re.compile(r"^[A-Za-z0-9_-]{1,40}/[A-Za-z0-9_.-]{1,80}$")
MAX_PROBE = 8 * 1024 * 1024  # Bytes; 30 s 48 kHz Mono 16 bit sind ~2,9 MB
LAUFEND = collections.Counter()           # gleichzeitige Sprech-Anfragen (für /health)
LAUFEND_SPERRE = threading.Lock()


def bekannt(stimme):
    return stimme in M.frontend.spk2info


class Hoch2:
    """24 -> 48 kHz über Blockgrenzen hinweg ohne Einschwinger (Filterzustand bleibt)."""

    def __init__(self):
        self.h = firwin(63, 0.5) * 2
        self.zi = np.zeros(len(self.h) - 1)

    def __call__(self, x):
        auf = np.zeros(2 * len(x))
        auf[::2] = x
        y, self.zi = lfilter(self.h, 1.0, auf, zi=self.zi)
        return y


def kopf(rate):
    p = 0x7FFFFFFF
    return (b"RIFF" + struct.pack("<I", p) + b"WAVE" + b"fmt "
            + struct.pack("<IHHIIHH", 16, 1, 1, rate, rate * 2, 2, 16) + b"data" + struct.pack("<I", p))


def pcm16(x):
    return (np.clip(x, -1.0, 1.0) * 32767).astype("<i2").tobytes()


class H(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"

    def _json(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        if self.path == "/health":
            return self._json(200, {"ok": True, "stimmen": STIMMEN, "rate": SR, "laufend": LAUFEND["n"],
                                    "llm_gpu": str(LLM_GPU), "flow_gpu": str(FLOW_GPU), "schritte": SCHRITTE,
                                    "graph": GD is not None})
        if self.path.startswith("/v1/models"):
            return self._json(200, {"data": [{"id": MODELL_NAME, "object": "model"}]})
        if self.path.startswith("/v1/voices"):
            return self._json(200, {"data": [{"id": v} for v in sorted(M.frontend.spk2info)]})
        self._json(404, {"error": "unbekannt"})

    def _stimme_anlegen(self, body):
        kennung = str(body.get("id", ""))
        text = str(body.get("text", "")).strip()
        if not KENNUNG.match(kennung) or not text:
            return self._json(400, {"error": "id (namensraum/name) und text noetig"})
        if bekannt(kennung):
            return self._json(200, {"ok": True, "id": kennung, "neu": False})
        try:
            wav = base64.b64decode(body.get("wav_b64", ""), validate=True)
        except Exception:
            return self._json(400, {"error": "wav_b64 kaputt"})
        if not wav.startswith(b"RIFF") or len(wav) > MAX_PROBE:
            return self._json(400, {"error": f"keine WAV oder groesser als {MAX_PROBE} Bytes"})
        t = time.time()
        with tempfile.NamedTemporaryFile(suffix=".wav") as f:
            f.write(wav)
            f.flush()
            try:
                stimme_laden(text, f.name, kennung)
            except Exception as e:
                print(f"Stimme {kennung} abgelehnt: {e}", flush=True)
                return self._json(400, {"error": f"Probe nicht lesbar: {e}"})
        print(json.dumps({"stimme_angelegt": kennung, "bytes": len(wav),
                          "dauer_s": round(time.time() - t, 2)}), flush=True)
        return self._json(200, {"ok": True, "id": kennung, "neu": True})

    def do_POST(self):
        if self.path not in ("/v1/audio/speech", "/v1/voices"):
            return self._json(404, {"error": "unbekannt"})
        laenge = int(self.headers.get("Content-Length", 0))
        if laenge > 2 * MAX_PROBE:
            return self._json(413, {"error": "zu gross"})
        try:
            body = json.loads(self.rfile.read(laenge))
        except Exception:
            return self._json(400, {"error": "kein JSON"})
        if self.path == "/v1/voices":
            return self._stimme_anlegen(body)
        text = str(body.get("input", "")).strip()
        stimme = body.get("voice") or STIMMEN[0]
        rate = int(body.get("sample_rate") or SR)
        try:
            tempo = min(2.0, max(0.5, float(body.get("speed") or 1.0)))
        except (TypeError, ValueError):
            return self._json(400, {"error": "speed keine Zahl"})
        strom = abs(tempo - 1.0) < 0.01       # Tempo nur ohne Streaming (CosyVoice-Grenze)
        if not text:
            return self._json(400, {"error": "input leer"})
        if not bekannt(stimme):
            return self._json(404, {"error": f"Stimme {stimme} unbekannt"})
        if body.get("response_format", "wav") != "wav" or rate not in (SR, 2 * SR):
            return self._json(400, {"error": f"nur wav mit {SR} oder {2 * SR} Hz"})
        t = time.time()
        with LAUFEND_SPERRE:
            LAUFEND["n"] += 1
            gleichzeitig = LAUFEND["n"]
        self.send_response(200)
        self.send_header("Content-Type", "audio/wav")
        self.end_headers()
        hoch = Hoch2() if rate == 2 * SR else None
        erst = None
        audio_s = 0.0
        try:
            self.wfile.write(kopf(rate))
            m.token_hop_len = HOP
            for o in M.inference_zero_shot(text, "", "", zero_shot_spk_id=stimme, stream=strom,
                                           speed=1.0 if strom else tempo):
                x = o["tts_speech"].squeeze(0).float().cpu().numpy()
                audio_s += len(x) / SR
                if hoch is not None:
                    x = hoch(x)
                self.wfile.write(pcm16(x))
                self.wfile.flush()
                if erst is None:
                    erst = time.time() - t
        except (BrokenPipeError, ConnectionResetError):
            print("Abnehmer weg, Satz abgebrochen", flush=True)
        except Exception as e:
            print(f"FEHLER mitten im Satz ({len(text)} Zeichen, {audio_s:.1f} s gesprochen): {e!r}", flush=True)
        finally:
            with LAUFEND_SPERRE:
                LAUFEND["n"] -= 1
        dauer = time.time() - t
        print(json.dumps({"stimme": stimme, "zeichen": len(text), "gleichzeitig": gleichzeitig,
                          "erst_s": round(erst or -1, 2), "gesamt_s": round(dauer, 2),
                          "audio_s": round(audio_s, 2), "rtf": round(dauer / audio_s, 2) if audio_s else None,
                          "rate": rate, "tempo": tempo}), flush=True)

    def log_message(self, *a):
        pass


srv = http.server.ThreadingHTTPServer(("0.0.0.0", PORT), H)
srv.daemon_threads = True
srv.serve_forever()
