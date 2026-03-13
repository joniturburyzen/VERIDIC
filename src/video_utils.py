# src/video_utils.py
# Extraccion de frames y audio de un video WebM/MP4 via PyAV
import io
import numpy as np


def extract_frames(video_bytes: bytes, max_frames: int = 120):
    """
    Extrae hasta max_frames frames del video como arrays BGR.
    Devuelve (frames_list, fps_efectivo).
    """
    import av

    container = av.open(io.BytesIO(video_bytes))
    v_stream  = container.streams.video[0]

    # fps real del stream
    avg_rate = v_stream.average_rate
    fps_real = float(avg_rate) if avg_rate else 30.0

    # Total de frames aproximado (puede ser 0 en WebM streaming)
    total = v_stream.frames or 0
    step  = max(1, total // max_frames) if total > 0 else 3

    frames = []
    for i, frame in enumerate(container.decode(video=0)):
        if i % step == 0:
            img = frame.to_ndarray(format="bgr24")
            frames.append(img)
            if len(frames) >= max_frames:
                break

    container.close()

    fps_eff = fps_real / step if step > 0 else fps_real
    return frames, fps_eff


def extract_audio(video_bytes: bytes, target_sr: int = 16000):
    """
    Extrae la pista de audio del video y la convierte a float32 mono.
    Devuelve (audio_array, sample_rate). Si no hay audio devuelve (None, target_sr).
    Compatible con PyAV antiguo y nuevo (resample() puede devolver frame o iterable).
    """
    import av

    container = av.open(io.BytesIO(video_bytes))

    # Buscar streams de audio de forma robusta
    audio_streams = [s for s in container.streams if s.type == 'audio']
    if not audio_streams:
        container.close()
        print("[audio] no audio stream found in video")
        return None, target_sr

    resampler = av.AudioResampler(format="fltp", layout="mono", rate=target_sr)
    chunks = []

    def _consume(result):
        """Acepta frame unico, lista o iterable segun version de PyAV."""
        if result is None:
            return
        if hasattr(result, 'to_ndarray'):
            arr = result.to_ndarray()
            chunks.append(arr[0] if arr.ndim > 1 else arr)
        else:
            for frame in result:
                if frame is not None:
                    arr = frame.to_ndarray()
                    chunks.append(arr[0] if arr.ndim > 1 else arr)

    try:
        for frame in container.decode(audio=0):
            _consume(resampler.resample(frame))
        _consume(resampler.resample(None))  # flush
    except Exception as e:
        print(f"[audio] extraction error: {e}")

    container.close()

    if not chunks:
        print("[audio] chunks empty after decode")
        return None, target_sr

    audio = np.concatenate(chunks).astype(np.float32)
    print(f"[audio] extracted {len(audio)/target_sr:.1f}s at {target_sr}Hz")
    return audio, target_sr
