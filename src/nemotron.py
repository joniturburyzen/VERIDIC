# src/nemotron.py
# Sintesis final via Nemotron 4 340B a traves del Cloudflare Worker.

import os
import json
import urllib.request
import urllib.error

_WORKER_URL = os.environ.get(
    "WORKER_URL", "https://empty-silence-364e.joniturbu.workers.dev"
).rstrip("/")


def synthesize(
    score:               float,
    features:            dict,
    transcription:       str,
    visual_analysis:     str,
    linguistic_analysis: str,
) -> str:
    """
    Llama a /synthesize en el Worker → Nemotron 4 340B.
    Devuelve el veredicto razonado como texto o "".
    """
    payload = {
        "score":               score,
        "features":            features,
        "transcription":       transcription       or "",
        "visual_analysis":     visual_analysis     or "",
        "linguistic_analysis": linguistic_analysis or "",
    }
    data = json.dumps(payload).encode()
    req  = urllib.request.Request(
        f"{_WORKER_URL}/synthesize",
        data=data,
        headers={
            "Content-Type": "application/json",
            "User-Agent":   "Mozilla/5.0 (compatible; PolygraphApp/1.0)",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=90) as resp:
            result = json.loads(resp.read())
            return result.get("text", "")
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        print(f"[nemotron] HTTP {e.code}: {body}")
        return ""
    except Exception as e:
        print(f"[nemotron] error: {e}")
        return ""
