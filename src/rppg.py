# src/rppg.py
# rPPG: estimacion de frecuencia cardiaca desde la camara (canal verde de frente)
import numpy as np
from scipy import signal as sp_signal


def forehead_green(frame: np.ndarray, lm: np.ndarray) -> float:
    """
    Extrae el valor medio del canal verde en la region de la frente.
    frame: BGR numpy array. lm: (478,2) landmarks normalizados.
    """
    h, w = frame.shape[:2]

    # Puntos de referencia: frente alta (10), esquinas faciales (234, 454), nariz (1)
    top_y   = int(lm[10][1]  * h)
    nose_y  = int(lm[1][1]   * h)
    left_x  = int(lm[234][0] * w)
    right_x = int(lm[454][0] * w)

    # ROI: frente, 20-70% del espacio vertical entre frente y nariz
    roi_y1 = top_y  + int((nose_y - top_y) * 0.20)
    roi_y2 = top_y  + int((nose_y - top_y) * 0.55)
    roi_x1 = left_x + int((right_x - left_x) * 0.15)
    roi_x2 = right_x - int((right_x - left_x) * 0.15)

    roi_y1, roi_y2 = max(0, roi_y1), min(h, roi_y2)
    roi_x1, roi_x2 = max(0, roi_x1), min(w, roi_x2)

    if roi_y2 <= roi_y1 or roi_x2 <= roi_x1:
        return 0.0

    roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]
    return float(roi[:, :, 1].mean())  # canal G de BGR


def compute_hr(green_signal: list, fps: float):
    """
    Estima HR (bpm) y variabilidad a partir de la senal verde.
    Devuelve (hr_bpm, hr_std). Si la senal es insuficiente devuelve (0, 0).
    """
    if len(green_signal) < 30:
        return 0.0, 0.0

    sig = np.array(green_signal, dtype=np.float64)
    sig = sp_signal.detrend(sig)
    sig = (sig - sig.mean()) / (sig.std() + 1e-8)

    nyq  = fps / 2.0
    low  = 0.7  / nyq   # ~42 bpm
    high = min(3.5 / nyq, 0.99)  # ~210 bpm

    b, a = sp_signal.butter(3, [low, high], btype='band')
    try:
        filtered = sp_signal.filtfilt(b, a, sig)
    except Exception:
        return 0.0, 0.0

    freqs    = np.fft.rfftfreq(len(filtered), d=1.0 / fps)
    fft_vals = np.abs(np.fft.rfft(filtered))
    valid    = (freqs >= 0.7) & (freqs <= 3.5)

    if not np.any(valid):
        return 0.0, 0.0

    hr_bpm = float(freqs[valid][np.argmax(fft_vals[valid])] * 60.0)

    # HRV proxy: std de intervalos entre picos del rPPG filtrado
    try:
        from scipy.signal import find_peaks
        min_dist = max(1, int(fps * 0.4))  # minimo ~40 bpm entre picos
        peaks, _ = find_peaks(filtered, distance=min_dist)
        if len(peaks) >= 3:
            rr_bpm = np.diff(peaks) / fps * 60.0
            hr_std = float(np.std(rr_bpm))
        else:
            hr_std = 0.0
    except Exception:
        hr_std = 0.0

    return hr_bpm, hr_std
