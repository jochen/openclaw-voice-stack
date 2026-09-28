"""laya-serve mit einem EIGENEN Checkpoint (nicht dem vom Hub).

laya-serve selbst kennt nur die veroeffentlichten Checkpoints. Der Aktuator-
Checkpoint ist auf die Ziele eines Hauses trainiert und liegt deshalb nicht
im Image, sondern wird eingehaengt (LAYA_CKPT). Er wird unter dem Namen
"multilingual" angeboten — so fragt der Voice-Assistant
(voice_assistant/services/laya_intent.py).
"""
import os

import uvicorn
from laya.router import Router
from laya.serve import create_app

ckpt = os.environ["LAYA_CKPT"]
router = Router(models={"multilingual": (ckpt, None)},
                device=os.environ.get("LAYA_DEVICE") or None, default="multilingual")
router.preload(["multilingual"])
uvicorn.run(create_app(router), host="0.0.0.0", port=int(os.environ.get("LAYA_PORT", "8096")),
            log_level="warning")
