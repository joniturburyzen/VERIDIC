# src/groq_analysis.py
# Llama directamente a Groq API y Google AI API.
# Requiere secretos: GROQ_KEY, GOOGLE_AI_API_KEY

import os, io, json, base64, uuid
import urllib.request, urllib.error

import cv2
import numpy as np
import soundfile as sf

_GROQ_KEY   = os.environ.get("GROQ_KEY", "")
_GOOGLE_KEY = os.environ.get("GOOGLE_AI_API_KEY", "")


# ── HTTP helpers ──────────────────────────────────────────────────────────────

def _groq_json(path: str, payload: dict) -> dict:
    url  = f"https://api.groq.com{path}"
    data = json.dumps(payload).encode()
    req  = urllib.request.Request(url, data=data, headers={
        "Content-Type":  "application/json",
        "Authorization": f"Bearer {_GROQ_KEY}",
    }, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        print(f"[groq] {path} HTTP {e.code}: {e.read().decode(errors='replace')[:300]}")
        return {}
    except Exception as e:
        print(f"[groq] {path} error: {e}")
        return {}


def _groq_multipart(path: str, fields: list, files: list) -> dict:
    """POST multipart/form-data a Groq sin dependencias externas."""
    boundary = uuid.uuid4().hex
    body = b""
    for name, value in fields:
        body += (f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n').encode()
    for name, filename, ctype, data in files:
        body += (f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"; filename="{filename}"\r\nContent-Type: {ctype}\r\n\r\n').encode()
        body += data + b"\r\n"
    body += f"--{boundary}--\r\n".encode()

    url = f"https://api.groq.com{path}"
    req = urllib.request.Request(url, data=body, headers={
        "Content-Type":  f"multipart/form-data; boundary={boundary}",
        "Authorization": f"Bearer {_GROQ_KEY}",
    }, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        print(f"[groq-mp] {path} HTTP {e.code}: {e.read().decode(errors='replace')[:300]}")
        return {}
    except Exception as e:
        print(f"[groq-mp] {path} error: {e}")
        return {}


def _gemini_json(payload: dict) -> dict:
    url  = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={_GOOGLE_KEY}"
    data = json.dumps(payload).encode()
    req  = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        print(f"[gemini] HTTP {e.code}: {e.read().decode(errors='replace')[:300]}")
        return {}
    except Exception as e:
        print(f"[gemini] error: {e}")
        return {}


# ── helpers de conversion ─────────────────────────────────────────────────────

def _audio_to_wav_bytes(audio: np.ndarray, sr: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV")
    return buf.getvalue()


def _frame_to_b64(frame: np.ndarray, quality: int = 72) -> str:
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf.tobytes()).decode()


# ── API publica ───────────────────────────────────────────────────────────────

def transcribe_audio(audio: np.ndarray, sr: int) -> dict:
    if audio is None or len(audio) < sr * 1.0:
        return {}
    wav = _audio_to_wav_bytes(audio, sr)
    d   = _groq_multipart(
        "/openai/v1/audio/transcriptions",
        fields=[
            ("model",                     "whisper-large-v3"),
            ("response_format",           "verbose_json"),
            ("timestamp_granularities[]", "word"),
        ],
        files=[("file", "audio.wav", "audio/wav", wav)],
    )
    if not d:
        return {}
    return {
        "text":  d.get("text", ""),
        "words": [{"word": w["word"], "start": w["start"], "end": w["end"]}
                  for w in d.get("words", [])],
    }


def analyze_key_frames(frames: list, video_bytes: bytes = None) -> str:
    if not frames and not video_bytes:
        return ""

    # Prioridad: Gemini 2.5 Flash con video completo
    if video_bytes and _GOOGLE_KEY:
        video_b64 = base64.b64encode(video_bytes).decode()
        resp = _gemini_json({
            "contents": [{"parts": [
                {"inline_data": {"mime_type": "video/webm", "data": video_b64}},
                {"text": (
                    "Analiza esta grabacion de video buscando señales no verbales de engano. "
                    "Observa: asimetria facial, micro-expresiones fugaces, tension en cejas y mandibula, "
                    "cambios en apertura ocular, movimientos de cabeza, incongruencia entre expresion y discurso. "
                    "Describe los hallazgos mas relevantes en 3-4 frases concisas en espanol sin asteriscos. "
                    "Al final añade exactamente una linea con el formato: "
                    "APARIENCIA: [edad aproximada, cabello, rasgos fisicos evidentes]. "
                    "Si no puedes determinarlo con certeza escribe: APARIENCIA: no determinable."
                )},
            ]}],
            "generationConfig": {"maxOutputTokens": 400, "temperature": 0.2},
        })
        text = ""
        try:
            text = resp["candidates"][0]["content"]["parts"][0]["text"].strip()
        except Exception:
            pass
        if text:
            return text

    # Fallback: Llama 4 Scout Vision con 3 frames
    if not frames:
        return ""

    labels  = ["Frame baseline (expresion neutra)", "Frame maxima actividad facial 1", "Frame maxima actividad facial 2"]
    content = []
    for i, frame in enumerate(frames[:3]):
        content.append({"type": "text", "text": f"**{labels[i]}:**"})
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{_frame_to_b64(frame)}"}})
    content.append({"type": "text", "text": (
        "El primer frame es la expresion neutra de referencia. "
        "Los frames 2 y 3 son los momentos de mayor actividad facial detectada. "
        "Compara los frames 2 y 3 con el baseline: busca asimetria facial, "
        "tension muscular (cejas, mandibula, boca), cambios en apertura ocular, "
        "micro-expresiones fugaces, diferencias entre lado izquierdo y derecho. "
        "Responde en 3-4 frases concisas en espanol sin asteriscos. "
        "Al final añade exactamente una linea con el formato: "
        "APARIENCIA: [edad aproximada, cabello, rasgos fisicos evidentes]. "
        "Si no puedes determinarlo con certeza escribe: APARIENCIA: no determinable."
    )})

    d = _groq_json("/openai/v1/chat/completions", {
        "model":      "meta-llama/llama-4-scout-17b-16e-instruct",
        "messages":   [{"role": "user", "content": content}],
        "max_tokens": 300,
    })
    return (d.get("choices") or [{}])[0].get("message", {}).get("content", "").strip()


# ── Linguistica ───────────────────────────────────────────────────────────────

_LING_SYS = (
    "Eres un experto en linguistica forense. Aplicas dos frameworks cientificos validados:\n\n"
    "CBCA — Criteria-Based Content Analysis (Steller & Köhnken 1989):\n"
    "Las declaraciones VERDADERAS tienden a presentar mas:\n"
    "  + Detalles contextuales ricos (lugar, tiempo, objetos, personas presentes)\n"
    "  + Produccion no estructurada: cronologia irregular, saltos naturales de memoria\n"
    "  + Complicaciones inesperadas mencionadas sin razon aparente\n"
    "  + Detalles superfluos irrelevantes para la historia (tipico de memoria real)\n"
    "  + Correcciones espontaneas (\"bueno, en realidad era...\")\n"
    "  + Admision de lagunas de memoria (\"no recuerdo si llevaba o no...\")\n"
    "  + Descripcion del estado mental propio o del interlocutor\n"
    "Las declaraciones FALSAS tienden a ser: logicamente perfectas, escasas en detalle, "
    "sin contradicciones ni lagunas, con estructura narrativa demasiado limpia.\n\n"
    "REALITY MONITORING (Johnson & Raye 1981 / Sporer 2004):\n"
    "Memorias REALES → ricas en informacion perceptual (visual, auditiva, tactil, olfativa), "
    "espacial y temporal especifica.\n"
    "Memorias FABRICADAS → mas operaciones cognitivas: justificaciones, inferencias, "
    "\"creo que\", \"supongo\", \"probablemente\".\n\n"
    "ALERTAS DE ENGANO adicionales:\n"
    "  - Distanciamiento pronominal: evita \"yo\", usa impersonal o tercera persona\n"
    "  - Over-explanation: justifica sin que se le pregunte\n"
    "  - Vaguedad temporal: \"creo que fue\", \"no recuerdo exactamente cuando\"\n"
    "  - Repeticion de negaciones: \"no menti\", \"nunca haria eso\"\n"
    "  - Respuestas que no responden la pregunta directamente\n\n"
    "FORMATO OBLIGATORIO (siempre en espanol, sin asteriscos de markdown):\n"
    "CBCA: [2-3 criterios observados o ausentes, con cita textual breve si es posible]\n"
    "REALITY MONITORING: [evaluacion del ratio perceptual vs cognitivo, 1-2 frases]\n"
    "ALERTAS: [indicadores de engano detectados; si no hay ninguno, escribe \"ninguno\"]\n"
    "RIESGO LINGUISTICO: BAJO / MEDIO / ALTO"
)

_FILLERS = {
    'eh','uh','um','ah','mm','hmm','hm','er',
    'bueno','pues','mira','oye','vamos','venga','claro',
    'o','sea','osea','eso','esto','entonces','es que','a ver',
    'basically','like','you know','i mean','right','so','well',
}


def _detect_fillers(words: list) -> dict:
    if not words:
        return {"count": 0, "rate": 0.0, "examples": []}
    found = []
    for i, w in enumerate(words):
        word   = (w.get("word") or "").strip().lower().rstrip(".,;:!?¿¡")
        bigram = word + " " + (words[i+1].get("word") or "").strip().lower() if i < len(words)-1 else ""
        if word in _FILLERS or bigram in _FILLERS:
            pause = (words[i+1].get("start", 0) - w.get("end", 0)) if i < len(words)-1 else 0
            found.append({"word": w.get("word",""), "pauseAfter": pause})
    dur_min  = ((words[-1].get("end",0) - words[0].get("start",0)) / 60) if len(words) > 1 else 1
    rate     = len(found) / max(dur_min, 1/60)
    examples = [f'"{f["word"]}" (+{f["pauseAfter"]:.1f}s)' for f in found if f["pauseAfter"] > 0.4][:6]
    return {"count": len(found), "rate": round(rate * 10) / 10, "examples": examples}


def analyze_linguistics(transcript_text: str, words: list) -> dict:
    if not transcript_text.strip():
        return {}

    pauses    = []
    if words and len(words) > 1:
        for i in range(1, len(words)):
            gap = (words[i].get("start", 0) - words[i-1].get("end", 0))
            if gap > 0.8:
                pauses.append(f'{gap:.1f}s antes de "{words[i].get("word","")}"')

    fillers   = _detect_fillers(words)
    word_list = transcript_text.strip().split()
    wc        = max(len(word_list), 1)
    pron_i    = sum(1 for w in word_list if w.lower() in {"yo","me","mi","mio","conmigo"})
    hedges    = sum(1 for w in word_list if w.lower() in {"creo","supongo","quizas","probablemente","talvez"})

    context = (
        f"Palabras totales: {wc}\n"
        f"Pronombre \"yo\" y variantes: {pron_i} (ratio {pron_i/wc*100:.1f}%)\n"
        f"Expresiones de incertidumbre cognitiva: {hedges}\n"
        f"Muletillas/hesitaciones: {fillers['count']} ({fillers['rate']}/min)"
        + (f" — con pausa: {', '.join(fillers['examples'])}" if fillers['examples'] else "") + "\n"
        + (f"Pausas >0.8s: {' | '.join(pauses[:8])}\n" if pauses else "")
    )

    d = _groq_json("/openai/v1/chat/completions", {
        "model":       "llama-3.3-70b-versatile",
        "messages":    [
            {"role": "system", "content": _LING_SYS},
            {"role": "user",   "content": f"METRICAS:\n{context}\nTRANSCRIPCION:\n{transcript_text}"},
        ],
        "max_tokens":  500,
        "temperature": 0.2,
    })
    text = (d.get("choices") or [{}])[0].get("message", {}).get("content", "").strip()
    return {"text": text, "filler_count": fillers["count"], "filler_rate": fillers["rate"]}
