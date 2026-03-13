# src/groq_analysis.py
# Llama al Cloudflare Worker que tiene las claves de Groq.
# El Worker expone: /transcribe  /visual  /linguistic

import os
import io
import json
import base64
import urllib.request
import urllib.error

import cv2
import numpy as np
import soundfile as sf

_WORKER_URL = os.environ.get(
    "WORKER_URL", "https://empty-silence-364e.joniturbu.workers.dev"
).rstrip("/")


# ── helper HTTP ───────────────────────────────────────────────────────────────

def _post(path: str, payload: dict) -> dict:
    """POST JSON al Worker. Devuelve dict o {} si falla."""
    url  = f"{_WORKER_URL}{path}"
    data = json.dumps(payload).encode()
    req  = urllib.request.Request(
        url, data=data,
        headers={
            "Content-Type": "application/json",
            "User-Agent":   "Mozilla/5.0 (compatible; PolygraphApp/1.0)",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        print(f"[worker] {path} HTTP {e.code}: {body}")
        return {}
    except Exception as e:
        print(f"[worker] {path} error: {e}")
        return {}


# ── audio helpers ─────────────────────────────────────────────────────────────

def _audio_to_wav_b64(audio: np.ndarray, sr: int) -> str:
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV")
    return base64.b64encode(buf.getvalue()).decode()


def _frame_to_b64(frame: np.ndarray, quality: int = 72) -> str:
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf.tobytes()).decode()


# ── API publica ───────────────────────────────────────────────────────────────

def transcribe_audio(audio: np.ndarray, sr: int) -> dict:
    """
    Transcripcion de audio via Groq Whisper (a traves del Worker).
    Devuelve {"text": str, "words": [{word, start, end}]} o {}.
    """
    if audio is None or len(audio) < sr * 1.0:
        return {}
    audio_b64 = _audio_to_wav_b64(audio, sr)
    return _post("/transcribe", {"audio_b64": audio_b64})


def analyze_key_frames(frames: list, video_bytes: bytes = None) -> str:
    """
    Analisis visual via Gemini 2.0 Flash (video completo) con fallback
    a Llama 4 Scout Vision (3 frames clave). Devuelve texto o "".
    """
    if not frames and not video_bytes:
        return ""
    payload: dict = {}
    if frames:
        payload["frames_b64"] = [_frame_to_b64(f) for f in frames[:3]]
    if video_bytes:
        payload["video_b64"]  = base64.b64encode(video_bytes).decode()
        payload["mime_type"]  = "video/webm"
    result = _post("/visual", payload)
    return result.get("text", "")


def analyze_linguistics(transcript_text: str, words: list) -> dict:
    """
    Analisis linguistico de engano via Llama 3.3 70B (Worker).
    Devuelve {"text": str, "filler_count": int, "filler_rate": float} o {}.
    """
    if not transcript_text.strip():
        return {}
    return _post("/linguistic", {"transcript": transcript_text, "words": words})
