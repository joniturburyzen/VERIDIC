# src/groq_analysis.py
# Canal primario: Google AI (Gemini). Fallback: Groq.
# Secretos: GOOGLE_AI_API_KEY (primario), GROQ_KEY (fallback, opcional)

import os, io, json, base64, uuid
import urllib.request, urllib.error

import cv2
import numpy as np
import soundfile as sf

_GOOGLE_KEY = os.environ.get("GOOGLE_AI_API_KEY", "")
_GROQ_KEY   = os.environ.get("GROQ_KEY", "")


# ── HTTP helpers ──────────────────────────────────────────────────────────────

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


def _groq_text(system: str, user: str, model: str = "llama-3.3-70b-versatile",
               max_tokens: int = 500, temperature: float = 0.2) -> str:
    """Llamada de texto a Groq. Devuelve string vacío si GROQ_KEY no está configurada."""
    if not _GROQ_KEY:
        return ""
    data = json.dumps({"model": model, "messages": [
        {"role": "system", "content": system},
        {"role": "user",   "content": user},
    ], "max_tokens": max_tokens, "temperature": temperature}).encode()
    req = urllib.request.Request(
        "https://api.groq.com/openai/v1/chat/completions", data=data,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {_GROQ_KEY}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            d = json.loads(r.read())
            return (d.get("choices") or [{}])[0].get("message", {}).get("content", "").strip()
    except urllib.error.HTTPError as e:
        print(f"[groq-text] HTTP {e.code}: {e.read().decode(errors='replace')[:200]}")
        return ""
    except Exception as e:
        print(f"[groq-text] error: {e}")
        return ""


def _groq_multipart_audio(wav_bytes: bytes) -> str:
    """Transcripción de audio via Groq Whisper. Devuelve texto o string vacío."""
    if not _GROQ_KEY:
        return ""
    boundary = uuid.uuid4().hex
    body = (
        f'--{boundary}\r\nContent-Disposition: form-data; name="model"\r\n\r\nwhisper-large-v3\r\n'
        f'--{boundary}\r\nContent-Disposition: form-data; name="response_format"\r\n\r\njson\r\n'
        f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="audio.wav"\r\n'
        f'Content-Type: audio/wav\r\n\r\n'
    ).encode() + wav_bytes + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(
        "https://api.groq.com/openai/v1/audio/transcriptions", data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}",
                 "Authorization": f"Bearer {_GROQ_KEY}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read()).get("text", "").strip()
    except urllib.error.HTTPError as e:
        print(f"[groq-whisper] HTTP {e.code}: {e.read().decode(errors='replace')[:200]}")
        return ""
    except Exception as e:
        print(f"[groq-whisper] error: {e}")
        return ""


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
    wav   = _audio_to_wav_bytes(audio, sr)
    wav64 = base64.b64encode(wav).decode()
    resp  = _gemini_json({
        "contents": [{"parts": [
            {"inline_data": {"mime_type": "audio/wav", "data": wav64}},
            {"text": "Transcribe este audio con precision. Devuelve unicamente el texto transcrito, sin explicaciones ni formato adicional."},
        ]}],
        "generationConfig": {"maxOutputTokens": 1000, "temperature": 0.0},
    })
    text = ""
    try:
        text = resp["candidates"][0]["content"]["parts"][0]["text"].strip()
    except Exception:
        pass
    if not text:
        print("[transcribe] Gemini vacío, intentando Groq Whisper...")
        text = _groq_multipart_audio(wav)
    if not text:
        return {}
    return {"text": text, "words": []}


def analyze_key_frames(frames: list, video_bytes: bytes = None) -> str:
    if not frames and not video_bytes:
        return ""

    # Gemini 2.5 Flash con video completo
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

    # Fallback: Groq con 3 frames JPEG (sin video completo)
    if not frames or not _GROQ_KEY:
        return ""
    print("[vision] Gemini vacío, intentando Groq con frames JPEG...")
    labels  = ["expresion neutra (baseline)", "actividad facial maxima 1", "actividad facial maxima 2"]
    content = []
    for i, frame in enumerate(frames[:3]):
        content.append({"type": "text",      "text": f"{labels[i]}:"})
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{_frame_to_b64(frame)}"}})
    content.append({"type": "text", "text": (
        "Compara los frames 2 y 3 con el baseline. Busca asimetria facial, tension muscular, "
        "micro-expresiones, cambios en apertura ocular. "
        "3-4 frases en espanol sin asteriscos. "
        "Al final una linea: APARIENCIA: [edad aproximada, cabello, rasgos visibles] "
        "o APARIENCIA: no determinable."
    )})
    data = json.dumps({
        "model":       "meta-llama/llama-4-scout-17b-16e-instruct",
        "messages":    [{"role": "user", "content": content}],
        "max_tokens":  300,
        "temperature": 0.2,
    }).encode()
    req = urllib.request.Request(
        "https://api.groq.com/openai/v1/chat/completions", data=data,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {_GROQ_KEY}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            d = json.loads(r.read())
            return (d.get("choices") or [{}])[0].get("message", {}).get("content", "").strip()
    except Exception as e:
        print(f"[vision-groq] error: {e}")
        return ""


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

    resp = _gemini_json({
        "systemInstruction": {"parts": [{"text": _LING_SYS}]},
        "contents": [{"role": "user", "parts": [{"text": f"METRICAS:\n{context}\nTRANSCRIPCION:\n{transcript_text}"}]}],
        "generationConfig": {"maxOutputTokens": 500, "temperature": 0.2},
    })
    text = ""
    try:
        text = resp["candidates"][0]["content"]["parts"][0]["text"].strip()
    except Exception:
        pass
    if not text:
        print("[linguistics] Gemini vacío, intentando Groq...")
        text = _groq_text(
            system=_LING_SYS,
            user=f"METRICAS:\n{context}\nTRANSCRIPCION:\n{transcript_text}",
            max_tokens=500, temperature=0.2,
        )
    return {"text": text, "filler_count": fillers["count"], "filler_rate": fillers["rate"]}
