# src/face_capture.py
import os
import sys
import urllib.request
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# --------------------------------------------------------------
#  Modelo FaceLandmarker – se descarga una sola vez si no existe
# --------------------------------------------------------------
_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
)
_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "face_landmarker.task")
_MODEL_PATH = os.path.normpath(_MODEL_PATH)

if not os.path.exists(_MODEL_PATH):
    sys.stdout.buffer.write(b"[face_capture] Descargando modelo FaceLandmarker...\n")
    sys.stdout.buffer.flush()
    urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
    sys.stdout.buffer.write(b"[face_capture] Descarga completada.\n")
    sys.stdout.buffer.flush()

# --------------------------------------------------------------
#  FaceLandmarker object – se crea una sola vez
#  Usamos model_asset_buffer para evitar problemas de ruta en Windows
# --------------------------------------------------------------
with open(_MODEL_PATH, "rb") as _f:
    _model_buffer = _f.read()

_base_opts = mp_python.BaseOptions(model_asset_buffer=_model_buffer)
_face_opts = mp_vision.FaceLandmarkerOptions(
    base_options=_base_opts,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    output_face_blendshapes=True,
    output_facial_transformation_matrixes=False,
)
_landmarker = mp_vision.FaceLandmarker.create_from_options(_face_opts)


# --------------------------------------------------------------
#  Haar cascade para crop de cara – cargado desde ruta temporal
#  para evitar problemas con caracteres Unicode en la ruta del proyecto
# --------------------------------------------------------------
def _load_cascade():
    import shutil, tempfile
    cv2_data = os.path.join(os.path.dirname(cv2.__file__), "data")
    src = os.path.join(cv2_data, "haarcascade_frontalface_default.xml")
    if not os.path.exists(src):
        src = os.path.join(cv2_data, "haarcascade_frontalface_alt2.xml")
    tmp = tempfile.mkdtemp(prefix="haar_")
    dst = os.path.join(tmp, "face.xml")
    shutil.copy2(src, dst)
    cc = cv2.CascadeClassifier(dst)
    return cc if not cc.empty() else None

_cascade = _load_cascade()

def _crop_face(frame: np.ndarray, pad: float = 0.30, size: int = 480) -> np.ndarray:
    """
    Detecta la cara más grande y devuelve un crop cuadrado redimensionado.
    Si no detecta cara, devuelve el frame original.
    """
    if _cascade is None:
        return frame
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = _cascade.detectMultiScale(gray, scaleFactor=1.1,
                                      minNeighbors=5, minSize=(60, 60))
    if len(faces) == 0:
        return frame
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    fh, fw = frame.shape[:2]
    x1 = max(0, x - int(w * pad))
    y1 = max(0, y - int(h * pad))
    x2 = min(fw, x + w + int(w * pad))
    y2 = min(fh, y + h + int(h * pad))
    crop = frame[y1:y2, x1:x2]
    return cv2.resize(crop, (size, size))


def get_landmarks(frame: np.ndarray):
    """
    Devuelve (landmarks, blendshapes) donde
        * landmarks   → array (478, 2) con coordenadas normalizadas (x, y)
        * blendshapes → dict {nombre: score} con 52 blendshapes de MediaPipe
    Si no se detecta rostro, devuelve (None, None).
    Aplica crop de cara automático antes de MediaPipe para mayor precisión.
    """
    frame = _crop_face(frame)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    results = _landmarker.detect(mp_image)

    if not results.face_landmarks:
        return None, None

    # 478 puntos (468 malla + 10 iris) → x e y normalizados
    lm = np.array([[pt.x, pt.y] for pt in results.face_landmarks[0]])

    # 52 blendshapes → dict nombre: score [0-1]
    bs_dict = {}
    if results.face_blendshapes:
        bs_dict = {bs.category_name: bs.score for bs in results.face_blendshapes[0]}

    return lm, bs_dict
