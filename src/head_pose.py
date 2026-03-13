# src/head_pose.py
import cv2
import numpy as np

# 6 puntos 3D de una cara promedio (mm) – modelo estándar OpenCV
_MODEL_3D = np.array([
    [ 0.0,    0.0,    0.0],   # Nariz (lm 1)
    [ 0.0,  -63.6,  -12.5],   # Barbilla (lm 152)
    [-43.3,  32.7,  -26.0],   # Esquina ojo izq (lm 33)
    [ 43.3,  32.7,  -26.0],   # Esquina ojo der (lm 263)
    [-28.9, -28.9,  -24.1],   # Comisura boca izq (lm 61)
    [ 28.9, -28.9,  -24.1],   # Comisura boca der (lm 291)
], dtype=np.float64)

_LM_IDX = [1, 152, 33, 263, 61, 291]


def head_pose_euler(lm: np.ndarray, frame_w: int, frame_h: int):
    """
    Devuelve (roll, pitch, yaw) en grados a partir de los landmarks normalizados.
    lm: array (478, 2) con coordenadas normalizadas [0-1].
    """
    pts2d = np.array(
        [[lm[i][0] * frame_w, lm[i][1] * frame_h] for i in _LM_IDX],
        dtype=np.float64
    )

    focal = frame_w
    cx, cy = frame_w / 2.0, frame_h / 2.0
    cam = np.array([[focal, 0, cx],
                    [0, focal, cy],
                    [0,     0,  1]], dtype=np.float64)
    dist = np.zeros((4, 1))

    ok, rvec, _ = cv2.solvePnP(
        _MODEL_3D, pts2d, cam, dist, flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not ok:
        return 0.0, 0.0, 0.0

    rmat, _ = cv2.Rodrigues(rvec)
    sy = np.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2)

    if sy > 1e-6:
        roll  = float(np.degrees(np.arctan2( rmat[2, 1], rmat[2, 2])))
        pitch = float(np.degrees(np.arctan2(-rmat[2, 0], sy)))
        yaw   = float(np.degrees(np.arctan2( rmat[1, 0], rmat[0, 0])))
    else:
        roll  = float(np.degrees(np.arctan2(-rmat[1, 2], rmat[1, 1])))
        pitch = float(np.degrees(np.arctan2(-rmat[2, 0], sy)))
        yaw   = 0.0

    return roll, pitch, yaw
