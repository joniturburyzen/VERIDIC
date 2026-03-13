# src/emotion.py
# Deteccion de emociones desde blendshapes de MediaPipe (sin DeepFace).
# Basado en Action Units de Ekman mapeados a los 52 blendshapes de FaceLandmarker.

import numpy as np

_NEG_EMOTIONS = {"fear", "anger", "disgust", "sad"}


def emotion_from_blendshapes(bs: dict) -> str:
    """
    Devuelve la emocion dominante a partir del dict de blendshapes.
    Sin coste adicional: MediaPipe ya los calcula junto a los landmarks.
    """
    if not bs:
        return "neutral"

    scores = {
        "fear":    (bs.get("browInnerUp",       0) * 0.30 +
                    bs.get("eyeWideLeft",        0) * 0.20 +
                    bs.get("eyeWideRight",       0) * 0.20 +
                    bs.get("mouthStretchLeft",   0) * 0.15 +
                    bs.get("mouthStretchRight",  0) * 0.15),

        "anger":   (bs.get("browDownLeft",       0) * 0.35 +
                    bs.get("browDownRight",      0) * 0.35 +
                    bs.get("noseSneerLeft",      0) * 0.15 +
                    bs.get("noseSneerRight",     0) * 0.15),

        "disgust": (bs.get("noseSneerLeft",      0) * 0.35 +
                    bs.get("noseSneerRight",     0) * 0.35 +
                    bs.get("mouthFrownLeft",     0) * 0.15 +
                    bs.get("mouthFrownRight",    0) * 0.15),

        "sad":     (bs.get("browInnerUp",        0) * 0.20 +
                    bs.get("mouthFrownLeft",     0) * 0.40 +
                    bs.get("mouthFrownRight",    0) * 0.40),

        "happy":   (bs.get("mouthSmileLeft",     0) * 0.35 +
                    bs.get("mouthSmileRight",    0) * 0.35 +
                    bs.get("cheekSquintLeft",    0) * 0.15 +
                    bs.get("cheekSquintRight",   0) * 0.15),

        "surprise":(bs.get("browOuterUpLeft",    0) * 0.20 +
                    bs.get("browOuterUpRight",   0) * 0.20 +
                    bs.get("eyeWideLeft",        0) * 0.20 +
                    bs.get("eyeWideRight",       0) * 0.20 +
                    bs.get("jawOpen",            0) * 0.20),

        "neutral":  0.25,
    }
    return max(scores, key=scores.get)


def facs_deception_features(bs_list: list) -> dict:
    """
    Extrae 4 señales FACS de deception a partir de la secuencia de blendshapes.
    Todas son display-only (no van al XGBoost hasta tener más datos).

    - contempt_proxy:     asimetría de sonrisa (AU12 unilateral — señal de Ekman)
    - suppression_index:  cara demasiado quieta = supresión activa de expresión
    - emotion_variance:   fluctuación emocional total (alta = estrés interno)
    - au_fear_peak:       pico máximo de miedo en cualquier frame (capta microexpresiones)
    """
    if not bs_list:
        return {
            "contempt_proxy":    0.0,
            "suppression_index": 0.0,
            "emotion_variance":  0.0,
            "au_fear_peak":      0.0,
        }

    contempts   = []   # asimetría de sonrisa por frame
    activations = []   # activación emocional total por frame
    fear_scores = []   # score de miedo por frame

    for bs in bs_list:
        if not bs:
            continue

        # Contempt proxy: AU12 asimétrico (labio elevado un solo lado)
        smile_l = bs.get("mouthSmileLeft",  0.0)
        smile_r = bs.get("mouthSmileRight", 0.0)
        contempts.append(abs(smile_l - smile_r))

        # Activación total: suma de todos los blendshapes no-neutros significativos
        total = (
            bs.get("browInnerUp",       0.0) +
            bs.get("browDownLeft",      0.0) + bs.get("browDownRight",      0.0) +
            bs.get("eyeWideLeft",       0.0) + bs.get("eyeWideRight",       0.0) +
            bs.get("noseSneerLeft",     0.0) + bs.get("noseSneerRight",     0.0) +
            bs.get("mouthFrownLeft",    0.0) + bs.get("mouthFrownRight",    0.0) +
            bs.get("mouthStretchLeft",  0.0) + bs.get("mouthStretchRight",  0.0) +
            bs.get("jawOpen",           0.0) +
            smile_l + smile_r
        )
        activations.append(total)

        # Fear score: AU1 + AU5 bilateral
        fear = (
            bs.get("browInnerUp",   0.0) * 0.35 +
            bs.get("eyeWideLeft",   0.0) * 0.30 +
            bs.get("eyeWideRight",  0.0) * 0.30 +
            bs.get("mouthStretchLeft",  0.0) * 0.025 +
            bs.get("mouthStretchRight", 0.0) * 0.025
        )
        fear_scores.append(fear)

    if not activations:
        return {
            "contempt_proxy":    0.0,
            "suppression_index": 0.0,
            "emotion_variance":  0.0,
            "au_fear_peak":      0.0,
        }

    act_arr = np.array(activations)
    act_mean = float(act_arr.mean())
    act_std  = float(act_arr.std())

    # Suppression: cara muy quieta = std baja relativa a la media
    # Valor alto = muy quieto = posible supresión. Invertido para que sea intuitivo.
    suppression = 1.0 - min(1.0, act_std / (act_mean + 1e-8))

    return {
        "contempt_proxy":    round(float(np.mean(contempts)), 4),
        "suppression_index": round(suppression, 4),
        "emotion_variance":  round(act_std, 4),
        "au_fear_peak":      round(float(max(fear_scores)), 4),
    }
