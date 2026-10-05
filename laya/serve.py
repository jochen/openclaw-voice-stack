"""laya-serve mit einem EIGENEN Checkpoint (nicht dem vom Hub).

laya-serve selbst kennt nur die veroeffentlichten Checkpoints. Der Aktuator-
Checkpoint ist auf die Ziele eines Hauses trainiert und liegt deshalb nicht
im Image, sondern wird eingehaengt (LAYA_CKPT). Er wird unter dem Namen
"multilingual" angeboten — so fragt der Voice-Assistant
(voice_assistant/services/laya_intent.py).

/health traegt zusaetzlich `checkpoint`: Name des Verzeichnisses und die
capabilities-Version, fuer die er trainiert ist. Ein Checkpoint kennt nur die
Ziele seiner Version; aendern sie sich, antwortet er weiter, nur schlechter,
und kein Fehler weist darauf hin (2026-10-01 und 2026-10-05). Der Assistent
vergleicht diese Version mit der Live-Version (aktuator_schatten.pruefe_checkpoint),
das Nachtraining ebenso (tools/laya_nachtraining.py).
"""
import json
import os

import uvicorn
from laya.router import Router
from laya.serve import create_app

ckpt = os.environ["LAYA_CKPT"]
router = Router(models={"multilingual": (ckpt, None)},
                device=os.environ.get("LAYA_DEVICE") or None, default="multilingual")
router.preload(["multilingual"])
app = create_app(router)

try:
    with open(os.path.join(ckpt, "rl_agent_config.json"), encoding="utf-8") as f:
        _cfg = json.load(f)
except (OSError, ValueError):
    _cfg = {}
_checkpoint = {"name": os.path.basename(ckpt.rstrip("/")),
               "capabilities": _cfg.get("capabilities"),
               "seed": (_cfg.get("training") or {}).get("seed")}

# Die /health von laya-serve ersetzen statt eine eigene Route daneben: der
# Assistent und der Healthcheck fragen schon /health, und die Felder von
# laya-serve sollen unveraendert bleiben.
_alt = next(r for r in app.router.routes if getattr(r, "path", None) == "/health")
app.router.routes.remove(_alt)


@app.get("/health")
def health():
    d = _alt.endpoint()
    d["checkpoint"] = _checkpoint
    return d


uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("LAYA_PORT", "8096")),
            log_level="warning")
