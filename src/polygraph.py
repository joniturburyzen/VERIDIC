# src/polygraph.py
import joblib, numpy as np, os

_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "polygraph_xgb.pkl")
_model = joblib.load(_MODEL_PATH)

# Orden fijo que debe coincidir con el entrenamiento
FEATURE_NAMES = [
    "ear_mean",           # apertura media de ojos (menor = mas tenso)
    "ear_std",            # variabilidad de apertura
    "iris_ratio_mean",    # tamano relativo del iris
    "iris_ratio_std",     # variabilidad del iris
    "ear_asymmetry",      # diferencia izquierdo-derecho EAR (asimetria ocular)
    "brow_asymmetry",     # diferencia de elevacion de cejas
    "gaze_std",           # inestabilidad de mirada (std del centro de iris)
    "head_roll_std",      # variabilidad del giro lateral de cabeza
    "head_pitch_std",     # variabilidad de cabeceo
    "head_yaw_std",       # variabilidad de giro horizontal
    "hr_bpm",             # frecuencia cardiaca estimada (rPPG)
    "hr_std",             # variabilidad FC (HRV proxy)
    "pitch_mean",         # tono medio de voz (F0)
    "pitch_std",          # variabilidad de tono (jitter proxy)
    "energy_std",         # variabilidad de energia vocal (shimmer proxy)
    "pause_ratio",        # fraccion de silencio
    "speech_rate",        # fraccion de habla activa
    "suppression_index",  # FACS: supresion emocional (blendshapes asimetricos)
    "emotion_variance",   # FACS: varianza inter-emocional (expresion inconsistente)
    "au_fear_peak",       # FACS: pico de miedo/AU1+AU2+AU4 (microexpresiones)
]


def compute_score(features: dict) -> float:
    """Devuelve probabilidad de engano [0-1]."""
    X = np.array([[features.get(k, 0.0) for k in FEATURE_NAMES]])
    return float(_model.predict_proba(X)[0, 1])
