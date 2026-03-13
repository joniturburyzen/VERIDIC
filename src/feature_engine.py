# src/feature_engine.py
import numpy as np
from src.head_pose import head_pose_euler
from src.rppg import forehead_green, compute_hr
from src.emotion import emotion_from_blendshapes, facs_deception_features, _NEG_EMOTIONS


# ── EAR ────────────────────────────────────────────────────────────────────────
def eye_aspect_ratio(lm: np.ndarray, idx: list) -> float:
    """EAR clasico de 6 puntos."""
    p1, p2, p3, p4, p5, p6 = [lm[i] for i in idx]
    A = np.linalg.norm(p2 - p3)
    B = np.linalg.norm(p5 - p6)
    C = np.linalg.norm(p1 - p4)
    return float((A + B) / (2.0 * C + 1e-8))


def blink_rate_from_ears(ears: np.ndarray, fps: float) -> float:
    """Parpadeos por segundo con umbral dinamico: 75% del EAR basal individual."""
    threshold   = float(ears.mean() * 0.75)
    below       = ears < threshold
    transitions = np.diff(below.astype(int))
    n_blinks    = int(np.sum(transitions == 1))
    duration_s  = len(ears) / max(fps, 1.0)
    return n_blinks / max(duration_s, 1.0)


# ── IRIS ───────────────────────────────────────────────────────────────────────
def iris_size_ratio(lm: np.ndarray,
                    corner_a: int, corner_b: int,
                    iris_center: int, iris_perimeter: list) -> float:
    """
    Radio del iris normalizado por la anchura del ojo.
    Usa los landmarks 468-477 (iris) que devuelve FaceLandmarker.
    """
    center     = lm[iris_center]
    perim_pts  = [lm[i] for i in iris_perimeter]
    radius     = np.mean([np.linalg.norm(center - p) for p in perim_pts])
    eye_width  = np.linalg.norm(lm[corner_a] - lm[corner_b])
    return float(radius / (eye_width + 1e-8))


# ── PIPELINE TEMPORAL ──────────────────────────────────────────────────────────
def compute_video_features(
    frames:   list,
    lm_list:  list,
    bs_list:  list,   # blendshapes por frame (dict o None)
    fps:      float,
    audio=None,
    audio_sr: int = 16000,
) -> dict | None:
    """
    Agrega todas las features temporales de un video.
    Las emociones se derivan de los blendshapes de MediaPipe (sin DeepFace).
    Devuelve dict con FEATURE_NAMES o None si no hay suficientes datos.
    """
    from src.audio_features import extract_audio_features

    h, w = frames[0].shape[:2] if frames else (480, 640)

    ears, ear_ls, ear_rs       = [], [], []
    iris_ratios                = []
    rolls, pitches, yaws       = [], [], []
    green_signal               = []
    emotions                   = []
    gaze_xs, gaze_ys           = [], []
    brow_asym_frames           = []

    for frame, lm, bs in zip(frames, lm_list, bs_list):
        if lm is None:
            continue

        # EAR — mantener izquierdo/derecho separados para asimetría
        ear_l = eye_aspect_ratio(lm, [33, 160, 158, 133, 153, 144])
        ear_r = eye_aspect_ratio(lm, [362, 385, 387, 263, 373, 380])
        ears.append((ear_l + ear_r) / 2.0)
        ear_ls.append(ear_l)
        ear_rs.append(ear_r)

        # Iris (landmarks 468-477)
        ir_l = iris_size_ratio(lm, 33, 133,  468, [469, 470, 471, 472])
        ir_r = iris_size_ratio(lm, 362, 263, 473, [474, 475, 476, 477])
        iris_ratios.append((ir_l + ir_r) / 2.0)

        # Gaze instability — centro del iris en coords normalizadas
        # 468 = iris izquierdo centro, 473 = iris derecho centro
        gaze_xs.append((lm[468][0] + lm[473][0]) / 2.0)
        gaze_ys.append((lm[468][1] + lm[473][1]) / 2.0)

        # Brow asymmetry — altura de ceja vs párpado superior
        # 105 = pico ceja izquierda, 334 = pico ceja derecha
        # 159 = párpado superior izq, 386 = párpado superior der
        brow_lift_l = lm[159][1] - lm[105][1]   # positivo = ceja por encima del ojo
        brow_lift_r = lm[386][1] - lm[334][1]
        brow_asym_frames.append(abs(brow_lift_l - brow_lift_r))

        # Head pose
        r, p, y = head_pose_euler(lm, w, h)
        rolls.append(r); pitches.append(p); yaws.append(y)

        # rPPG: canal verde de la frente
        green_signal.append(forehead_green(frame, lm))

        # Emocion desde blendshapes (instantaneo, sin modelo externo)
        emotions.append(emotion_from_blendshapes(bs))

    if len(ears) < 10:
        return None  # datos insuficientes

    ears_arr        = np.array(ears)
    iris_arr        = np.array(iris_ratios)

    # -- Features faciales --
    ear_mean        = float(ears_arr.mean())
    ear_std         = float(ears_arr.std())
    blink_r         = float(blink_rate_from_ears(ears_arr, fps))
    iris_ratio_mean = float(iris_arr.mean())
    iris_ratio_std  = float(iris_arr.std())

    # -- Asimetría facial --
    ear_asymmetry   = float(np.mean(np.abs(np.array(ear_ls) - np.array(ear_rs))))
    brow_asymmetry  = float(np.mean(brow_asym_frames)) if brow_asym_frames else 0.0

    # -- Inestabilidad de mirada --
    gaze_std        = float(np.sqrt(np.std(gaze_xs)**2 + np.std(gaze_ys)**2))

    # -- Head pose -- unwrap para evitar saltos de fase ±180° que disparan el std
    head_roll_std   = float(np.degrees(np.std(np.unwrap(np.radians(rolls)))))
    head_pitch_std  = float(np.degrees(np.std(np.unwrap(np.radians(pitches)))))
    head_yaw_std    = float(np.degrees(np.std(np.unwrap(np.radians(yaws)))))

    # -- Emociones --
    if emotions:
        fear_rate    = sum(1 for e in emotions if e == "fear")       / len(emotions)
        neg_emo_rate = sum(1 for e in emotions if e in _NEG_EMOTIONS) / len(emotions)
    else:
        fear_rate = neg_emo_rate = 0.0

    # -- rPPG --
    hr_bpm, hr_std  = compute_hr(green_signal, fps)

    # -- FACS deception features (display-only, no van al XGBoost aún) --
    facs = facs_deception_features(bs_list)

    # -- Audio --
    aud_feats = extract_audio_features(audio, audio_sr)

    return {
        "ear_mean":         ear_mean,
        "ear_std":          ear_std,
        "blink_rate":       blink_r,
        "fear_rate":        fear_rate,
        "neg_emotion_rate": neg_emo_rate,
        "iris_ratio_mean":  iris_ratio_mean,
        "iris_ratio_std":   iris_ratio_std,
        "ear_asymmetry":    ear_asymmetry,
        "brow_asymmetry":   brow_asymmetry,
        "gaze_std":         gaze_std,
        "head_roll_std":    head_roll_std,
        "head_pitch_std":   head_pitch_std,
        "head_yaw_std":     head_yaw_std,
        "hr_bpm":           hr_bpm,
        "hr_std":           hr_std,
        "pitch_mean":       aud_feats["pitch_mean"],
        "pitch_std":        aud_feats["pitch_std"],
        "energy_std":       aud_feats["energy_std"],
        "pause_ratio":      aud_feats["pause_ratio"],
        "speech_rate":      aud_feats["speech_rate"],
        # FACS — display only
        "contempt_proxy":    facs["contempt_proxy"],
        "suppression_index": facs["suppression_index"],
        "emotion_variance":  facs["emotion_variance"],
        "au_fear_peak":      facs["au_fear_peak"],
    }
