# polygraph/api/endpoint.py
import io, os, asyncio
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
import cv2

from src.face_capture    import get_landmarks
from src.feature_engine  import compute_video_features
from src.polygraph       import compute_score, FEATURE_NAMES
from src.video_utils     import extract_frames, extract_audio
from src.groq_analysis   import transcribe_audio, analyze_key_frames, analyze_linguistics
from src.nemotron        import synthesize

app = FastAPI()

# ---- Frontend estatico ----
_FRONTEND = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "frontend"))
app.mount("/static", StaticFiles(directory=_FRONTEND), name="static")

@app.get("/")
def index():
    return FileResponse(os.path.join(_FRONTEND, "index.html"))

# ---- CORS ----
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
#  Pipeline sincrono (CPU-bound, corre en executor)
# ─────────────────────────────────────────────────────────────────────────────

# Features a comparar para calibración individual (baseline vs test)
_CALIB_FEATURES = [
    "ear_mean", "ear_std", "ear_asymmetry", "brow_asymmetry", "gaze_std",
    "iris_ratio_mean", "head_yaw_std", "head_pitch_std",
    "pitch_mean", "pause_ratio", "speech_rate",
]

def _compute_calibration(
    frames, lm_list, bs_list, fps, audio, audio_sr
) -> dict:
    """
    Divide el video en baseline (primeros 3s) y zona de analisis (resto).
    Devuelve dict con los deltas relativos: positivo = mas alto en analisis que en baseline.
    """
    baseline_frames = max(1, int(fps * 3))
    if len(frames) <= baseline_frames + 5:
        return {}   # video demasiado corto para calibrar

    def _feats(fr, lm, bs, aud):
        return compute_video_features(fr, lm, bs, fps, aud, audio_sr)

    # Segmentar landmarks/frames
    fr_base  = frames[:baseline_frames]
    lm_base  = lm_list[:baseline_frames]
    bs_base  = bs_list[:baseline_frames]

    fr_test  = frames[baseline_frames:]
    lm_test  = lm_list[baseline_frames:]
    bs_test  = bs_list[baseline_frames:]

    # Audio segmentado
    if audio is not None and audio_sr:
        t_cut   = baseline_frames / max(fps, 1)
        s_cut   = int(t_cut * audio_sr)
        aud_base = audio[:s_cut]
        aud_test = audio[s_cut:]
    else:
        aud_base = aud_test = None

    f_base = _feats(fr_base, lm_base, bs_base, aud_base)
    f_test = _feats(fr_test, lm_test, bs_test, aud_test)

    if f_base is None or f_test is None:
        return {}

    deltas = {}
    for k in _CALIB_FEATURES:
        base_val = f_base.get(k, 0.0)
        test_val = f_test.get(k, 0.0)
        if abs(base_val) > 1e-8:
            deltas[f"{k}_delta"] = round(float((test_val - base_val) / base_val), 4)

    return deltas


def _process_video(video_bytes: bytes) -> dict:
    """
    1. Extrae frames y audio
    2. MediaPipe → landmarks + blendshapes por frame
    3. feature_engine → 20 features (emociones desde blendshapes, sin DeepFace)
    4. XGBoost → score
    5. Calibración individual: deltas baseline vs zona de analisis
    6. Selecciona 3 frames clave para el analisis visual
    """
    frames, fps = extract_frames(video_bytes, max_frames=100)
    audio, audio_sr = extract_audio(video_bytes)

    if not frames:
        raise ValueError("No se pudieron extraer frames del video")

    lm_list, bs_list = [], []
    for frame in frames:
        lm, bs = get_landmarks(frame)
        lm_list.append(lm)
        bs_list.append(bs)

    valid = sum(1 for lm in lm_list if lm is not None)
    min_required = max(20, int(len(frames) * 0.30))
    if valid < min_required:
        raise ValueError(f"Rostro detectado en {valid}/{len(frames)} frames (mínimo {min_required}) — regrabar con mejor encuadre")

    features = compute_video_features(frames, lm_list, bs_list, fps, audio, audio_sr)
    if features is None:
        raise ValueError("No hay suficientes datos validos para el analisis")

    score = compute_score(features)

    # Calibración individual (no afecta al score del modelo)
    calib_deltas = _compute_calibration(frames, lm_list, bs_list, fps, audio, audio_sr)

    # Frames clave: baseline neutro + 2 momentos de maxima actividad facial
    valid_idx = [i for i, lm in enumerate(lm_list) if lm is not None]

    # Media de landmarks sobre todos los frames validos (expresion base)
    lm_valid = [lm_list[i] for i in valid_idx]
    mean_lm  = np.mean(np.stack(lm_valid), axis=0)   # (478, 2)

    # Distancia de cada frame respecto a la expresion media
    dists = [float(np.linalg.norm(lm_list[i] - mean_lm)) for i in valid_idx]

    # Frame 1: el mas calmado de los primeros 3s reales (zona baseline de calibración)
    baseline_frames_n = max(1, int(fps * 3))
    baseline_zone = [i for i in valid_idx if i < baseline_frames_n]
    if not baseline_zone:  # fallback si el vídeo es muy corto
        baseline_zone = valid_idx[:max(1, len(valid_idx) // 3)]
    dists_first = [float(np.linalg.norm(lm_list[i] - mean_lm)) for i in baseline_zone]
    baseline_idx = baseline_zone[int(np.argmin(dists_first))]

    # Frames 2 y 3: maxima desviacion, separados por al menos 10 frames entre si
    ranked = sorted(zip(dists, valid_idx), reverse=True)
    expressive = []
    for _, idx in ranked:
        if all(abs(idx - e) >= 10 for e in expressive) and idx != baseline_idx:
            expressive.append(idx)
        if len(expressive) == 2:
            break
    # Fallback si no hay suficientes frames distintos
    while len(expressive) < 2:
        expressive.append(valid_idx[len(valid_idx) // 2])

    key_frames = [frames[baseline_idx], frames[expressive[0]], frames[expressive[1]]]

    face_coverage = round(valid / max(len(frames), 1), 3)
    feat_display = {k: round(float(v), 4) for k, v in features.items()}
    feat_display.update(calib_deltas)

    return {
        "lie_probability": round(score, 4),
        "features":        feat_display,
        "_key_frames":     key_frames,
        "_audio":          audio,
        "_audio_sr":       audio_sr,
        "_calib_deltas":   calib_deltas,
        "_face_coverage":  face_coverage,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Endpoint principal
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/analyze_video")
async def analyze_video(file: UploadFile = File(...)):
    """
    Recibe un video corto (WebM/MP4) con audio.
    Pipeline:
      - Sincrono: MediaPipe + XGBoost  (executor)
      - Paralelo: Whisper + Vision     (Worker → Groq)
      - Secuencial: Linguistico        (Worker → Groq, necesita transcripcion)
      - Secuencial: Sintesis final     (Worker → Nemotron)
    """
    video_bytes = await file.read()
    loop = asyncio.get_event_loop()

    # ── Fase 1: pipeline CV + XGBoost ──────────────────────────────────────
    try:
        result = await loop.run_in_executor(None, _process_video, video_bytes)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error interno: {e}")

    key_frames    = result.pop("_key_frames")
    audio         = result.pop("_audio")
    audio_sr      = result.pop("_audio_sr")
    calib_deltas  = result.pop("_calib_deltas")
    face_coverage = result.pop("_face_coverage")

    _WORKER_TIMEOUT = 40.0   # segundos — timeout por llamada al Worker

    # ── Fase 2: Whisper + Vision en paralelo (I/O-bound via Worker) ────────
    t_task = loop.run_in_executor(None, transcribe_audio,   audio, audio_sr)
    v_task = loop.run_in_executor(None, analyze_key_frames, key_frames, video_bytes)

    raw_t, raw_v = await asyncio.gather(
        asyncio.wait_for(t_task, timeout=_WORKER_TIMEOUT),
        asyncio.wait_for(v_task, timeout=_WORKER_TIMEOUT),
        return_exceptions=True,
    )

    transcription_data = raw_t if isinstance(raw_t, dict) else {}
    visual_analysis    = raw_v if isinstance(raw_v, str)  else ""

    transcript_text = transcription_data.get("text",  "")
    words           = transcription_data.get("words", [])
    visual_analysis = visual_analysis or ""

    # ── Fase 3: Analisis linguistico (necesita transcripcion) ──────────────
    try:
        ling_result = await asyncio.wait_for(
            loop.run_in_executor(None, analyze_linguistics, transcript_text, words),
            timeout=_WORKER_TIMEOUT,
        )
    except Exception:
        ling_result = {}

    if not isinstance(ling_result, dict):
        ling_result = {}

    linguistic_analysis = ling_result.get("text", "")
    filler_count        = int(ling_result.get("filler_count", 0))
    filler_rate         = float(ling_result.get("filler_rate", 0.0))

    # ── Fase 4: Sintesis Nemotron (necesita todo) ───────────────────────────
    try:
        verdict = await asyncio.wait_for(
            loop.run_in_executor(
                None, synthesize,
                result["lie_probability"],
                result["features"],
                transcript_text,
                visual_analysis,
                linguistic_analysis or "",
                calib_deltas,
                face_coverage,
            ),
            timeout=_WORKER_TIMEOUT,
        )
    except Exception:
        verdict = ""

    # Añadir métricas de muletillas al grid de features (display only)
    if filler_count or filler_rate:
        result["features"]["filler_rate"]  = round(filler_rate, 1)
        result["features"]["filler_count"] = filler_count

    result.update({
        "transcription":       transcript_text,
        "visual_analysis":     visual_analysis,
        "linguistic_analysis": linguistic_analysis or "",
        "verdict":             verdict or "",
    })

    return JSONResponse(content=result)
