# src/audio_features.py
# Analisis de prosodia: pitch, energia, pausas, tasa de habla
import numpy as np

_EMPTY = {
    "pitch_mean": 0.0, "pitch_std": 0.0,
    "energy_mean": 0.0, "energy_std": 0.0,
    "pause_ratio": 0.0, "speech_rate": 0.0,
}


def extract_audio_features(audio: np.ndarray, sr: int) -> dict:
    """
    audio: float32 mono array.
    sr:    sample rate (Hz).
    Devuelve dict con 6 features de prosodia.
    """
    try:
        import librosa
    except ImportError:
        return _EMPTY.copy()

    if audio is None or len(audio) < sr * 0.5:
        return _EMPTY.copy()

    audio = audio.astype(np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=0)

    # -- Energia (RMS) --
    rms = librosa.feature.rms(y=audio, frame_length=2048, hop_length=512)[0]
    energy_mean = float(np.mean(rms))
    energy_std  = float(np.std(rms))

    # -- Pausas: frames con RMS < 15% de la media --
    silence_thresh = energy_mean * 0.15
    pause_ratio    = float(np.mean(rms < silence_thresh))

    # -- Tasa de habla: fraccion de frames con ZCR alto (voz activa) --
    zcr = librosa.feature.zero_crossing_rate(audio, frame_length=2048, hop_length=512)[0]
    speech_rate = float(np.mean(zcr > 0.05))

    # -- Pitch (F0) con YIN: mas rapido que pYIN --
    try:
        f0 = librosa.yin(audio, fmin=80, fmax=400, sr=sr, frame_length=2048)
        f0_valid = f0[(f0 > 85) & (f0 < 380)]
        pitch_mean = float(np.mean(f0_valid)) if len(f0_valid) > 0 else 0.0
        pitch_std  = float(np.std(f0_valid))  if len(f0_valid) > 0 else 0.0
    except Exception:
        pitch_mean, pitch_std = 0.0, 0.0

    return {
        "pitch_mean":  pitch_mean,
        "pitch_std":   pitch_std,
        "energy_mean": energy_mean,
        "energy_std":  energy_std,
        "pause_ratio": pause_ratio,
        "speech_rate": speech_rate,
    }
