# src/nemotron.py
# Sintesis final via Gemini 2.5 Flash (Google AI) — llamada directa.
# Requiere secreto: GOOGLE_AI_API_KEY

import os, json, re
import urllib.request, urllib.error

_GOOGLE_KEY = os.environ.get("GOOGLE_AI_API_KEY", "")

_SYNTHESIS_SYS = (
    "Eres un analista forense multimodal. Integras tres canales de evidencia "
    "para evaluar la autenticidad de una declaracion. Eres sensible: prefieres "
    "no pasar por alto una mentira que cometer un falso positivo.\n"
    "CONTEXTO: los videos analizados duran 8-15 segundos (muy cortos). Por ello:\n"
    "  - blink_rate=0 NO es significativo (necesita mas tiempo para ser fiable)\n"
    "  - fear_rate y neg_emotion_rate tienden a 0 en webcam casual → ignorarlos si son 0\n"
    "  - El canal LINGUISTICO es el mas fiable en videos cortos → darle mas peso\n"
    "  - suppression_index, au_fear_peak y emotion_variance SON fiables incluso en corto\n\n"
    "LOS TRES CANALES Y SU PESO RELATIVO:\n"
    "  [A] BIOMETRICO (40%): score XGBoost entrenado con dataset real de juicios.\n"
    "      Score >30% = señal moderada. Score >50% = señal fuerte. Score >70% = señal muy fuerte.\n"
    "      IMPORTANTE: el modelo esta entrenado en videos de sala judicial; en webcam\n"
    "      los scores suelen ser mas bajos de lo real, por lo que incluso un 20-35%\n"
    "      puede indicar señales relevantes si otros canales lo apoyan.\n"
    "  [B] LINGUISTICO (40%): analisis CBCA + Reality Monitoring.\n"
    "      RIESGO ALTO = señal fuerte de engano aunque el biometrico sea bajo.\n"
    "      RIESGO MEDIO = señal moderada que inclina hacia SOSPECHOSO en duda.\n"
    "      RIESGO BAJO = contraindica engano pero no lo descarta por si solo.\n"
    "  [C] VISUAL (20%): micro-expresiones y tension en video completo (Gemini).\n"
    "      Cualquier asimetria, tension o micro-expresion detectada es relevante.\n"
    "      El texto visual puede incluir una linea \"APARIENCIA:\" con descripcion fisica observable.\n"
    "      Si existe esa linea, cruzala con la transcripcion: si la persona declara rasgos fisicos\n"
    "      que contradicen lo visible, es una discrepancia factual.\n\n"
    "REGLAS DE INTEGRACION:\n"
    "  - RIESGO LINGUISTICO ALTO → veredicto SOSPECHOSO salvo que bio y visual sean claramente bajos\n"
    "  - RIESGO LINGUISTICO MEDIO + cualquier señal biometrica o visual → SOSPECHOSO o AMBIGUO\n"
    "  - Bio moderado (>30%) + visual con señales → AMBIGUO o SOSPECHOSO\n"
    "  - Discrepancia factual clara entre APARIENCIA y transcripcion → veredicto minimo SOSPECHOSO\n"
    "  - Solo si los tres canales apuntan a ausencia de señales → AUTENTICO\n"
    "  - En caso de duda, prefiere AMBIGUO a AUTENTICO\n\n"
    "FORMATO OBLIGATORIO (sin asteriscos ni markdown, en espanol):\n"
    "CANAL BIOMETRICO: [1-2 frases sobre las señales biometricas mas relevantes]\n"
    "CANAL LINGUISTICO: [1-2 frases resumiendo el riesgo linguistico]\n"
    "CANAL VISUAL: [1 frase]\n"
    "EVALUACION INTEGRADA: [2-3 frases de razonamiento conjunto]\n"
    "VEREDICTO: AUTENTICO / AMBIGUO / SOSPECHOSO — confianza: BAJA / MEDIA / ALTA\n"
    "PUNTUACION: [numero entero 0-100]\n"
    "REGLAS OBLIGATORIAS para PUNTUACION (NO son sugerencias, son limites duros):\n"
    "  - Linguistico ALTO → PUNTUACION minima 65, independientemente del bio\n"
    "  - Linguistico MEDIO → PUNTUACION minima 45\n"
    "  - Bio combinado >40% → sumar al menos 10 puntos extra\n"
    "  - suppression_index >0.5 → sumar al menos 8 puntos extra\n"
    "  - VEREDICTO SOSPECHOSO confianza ALTA → PUNTUACION minima 80\n"
    "  - VEREDICTO SOSPECHOSO confianza MEDIA → PUNTUACION minima 68\n"
    "  - VEREDICTO AMBIGUO confianza ALTA → PUNTUACION minima 55\n"
    "  - Discrepancia factual clara → sumar al menos 20 puntos\n"
    "  - Solo AUTENTICO con bio bajo Y linguistico BAJO → PUNTUACION puede ser <35"
)


def _gemini_json(payload: dict) -> dict:
    url  = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={_GOOGLE_KEY}"
    data = json.dumps(payload).encode()
    req  = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=90) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        print(f"[nemotron] HTTP {e.code}: {e.read().decode(errors='replace')[:300]}")
        return {}
    except Exception as e:
        print(f"[nemotron] error: {e}")
        return {}


def synthesize(
    score:               float,
    features:            dict,
    transcription:       str,
    visual_analysis:     str,
    linguistic_analysis: str,
) -> str:
    key_feats  = ['ear_mean','ear_asymmetry','gaze_std','brow_asymmetry','head_yaw_std',
                  'hr_bpm','pitch_mean','pause_ratio','speech_rate']
    facs_feats = ['contempt_proxy','suppression_index','emotion_variance','au_fear_peak']

    feat_summary = ", ".join(f"{k}={float(features[k]):.3f}" for k in key_feats if k in features)
    facs_summary = ", ".join(f"{k}={float(features[k]):.3f}" for k in facs_feats if k in features)

    m = re.search(r'RIESGO LINGUISTICO[^:]*:\s*(BAJO|MEDIO|ALTO)', linguistic_analysis or "", re.I)
    ling_risk = m.group(1) if m else ""

    # Bio score combinado: XGBoost calibrado + FACS
    xgb_raw     = float(score)
    xgb_scaled  = min(0.70, xgb_raw * 4.0)
    suppression = float(features.get("suppression_index", 0))
    au_fear     = float(features.get("au_fear_peak", 0))
    emo_var     = float(features.get("emotion_variance", 0))
    facs_signal = min(0.50,
        ((suppression - 0.35) * 0.8 if suppression > 0.35 else 0) +
        (au_fear * 0.4              if au_fear      > 0.08 else 0) +
        (emo_var * 0.2              if emo_var       > 0.08 else 0)
    )
    bio_score = round(min(1.0, xgb_scaled * 0.5 + facs_signal * 0.5) * 100)

    user_msg = (
        f"[A] BIOMETRICO\n"
        f"Score biometrico combinado (XGBoost calibrado + FACS): {bio_score}%\n"
        f"  (XGBoost webcam raw: {xgb_raw*100:.1f}% — automaticamente escalado x4 por diferencia de dominio)\n"
        f"Features biometricas: {feat_summary or '(no disponible)'}\n"
        + (f"Señales FACS (Ekman): {facs_summary}\n" if facs_summary else "")
        + "  suppression_index: >0.35=supresion activa, >0.55=supresion significativa, >0.75=supresion intensa.\n"
          "  au_fear_peak: >0.08=miedo presente, >0.15=miedo notable. contempt_proxy >0.03=desprecio.\n\n"
        f"[B] LINGUISTICO{f' — Riesgo: {ling_risk}' if ling_risk else ''}\n"
        f"{linguistic_analysis or '(no disponible)'}\n\n"
        f"[C] VISUAL\n"
        f"{visual_analysis or '(no disponible)'}\n\n"
        f"Transcripcion original: {transcription or '(no disponible)'}"
    )

    resp = _gemini_json({
        "systemInstruction": {"parts": [{"text": _SYNTHESIS_SYS}]},
        "contents": [{"role": "user", "parts": [{"text": user_msg}]}],
        "generationConfig": {"maxOutputTokens": 700, "temperature": 0.3},
    })
    text = ""
    try:
        text = resp["candidates"][0]["content"]["parts"][0]["text"].strip()
    except Exception:
        pass
    return text
