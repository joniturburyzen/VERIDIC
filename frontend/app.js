// app.js
import { FaceLandmarker, FilesetResolver, DrawingUtils }
  from 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs';

const video            = document.getElementById('video');
const meshCanvas       = document.getElementById('meshCanvas');
const btnRecord        = document.getElementById('btnRecord');
const recDot           = document.getElementById('recDot');
const countdown        = document.getElementById('countdown');
const statusText       = document.getElementById('statusText');
const statusBar        = document.getElementById('statusBar');
const results          = document.getElementById('results');
const ringFill         = document.getElementById('ringFill');
const scoreLabel       = document.getElementById('scoreLabel');
const verdict          = document.getElementById('verdict');
const verdictCard      = document.getElementById('verdictCard');
const verdictText      = document.getElementById('verdictText');
const analysisPanels   = document.getElementById('analysisPanels');
const transcriptionText= document.getElementById('transcriptionText');
const visualText       = document.getElementById('visualText');
const linguisticText   = document.getElementById('linguisticText');
const featureGrid      = document.getElementById('featureGrid');

// Screen system
const screenLoading    = document.getElementById('screenLoading');
const screenMain       = document.getElementById('screenMain');
const screenResults    = document.getElementById('screenResults');
const progressBar      = document.getElementById('progressBar');
const sysLogBody       = document.getElementById('sysLogBody');
const btnDetails       = document.getElementById('btnDetails');
const btnBack          = document.getElementById('btnBack');
const kernelBody       = document.getElementById('kernelBody');

const MAX_SEC = 15;
const MIN_SEC = 3;
const RING_C  = 314;

const lingBadge   = document.getElementById('lingBadge');
const livePanel   = document.getElementById('livePanel');
const liveEarBar  = document.getElementById('liveEarBar');
const liveEarVal  = document.getElementById('liveEarVal');
const liveGazeBar = document.getElementById('liveGazeBar');
const liveGazeVal = document.getElementById('liveGazeVal');
const liveHeadBar = document.getElementById('liveHeadBar');
const liveHeadVal = document.getElementById('liveHeadVal');

let mediaRecorder = null;
let chunks        = [];
let countdownId   = null;
let secsLeft      = MAX_SEC;

// ── Auto-zoom ─────────────────────────────────────────────────────────────────
let _autoZoom = 1.0;   // zoom calculado por posición de cara
let _autoTx   = 0;     // traslación X (% del elemento)
let _autoTy   = 0;     // traslación Y (% del elemento)
let _zoomMul  = 1.0;   // ajuste manual con botones +/-

// ── Grupos de landmarks para feedback de anomalías ────────────────────────────
const EYES_IDX = new Set([33,160,158,133,153,144,362,385,387,263,373,380]);
const IRIS_IDX = new Set([468,469,470,471,472,473,474,475,476,477]);

// Baseline EAR para detectar cierre anómalo de ojos durante grabación
const _earBaseline = [];

// ── Labels biometricos ────────────────────────────────────────────────────────
const FEAT_LABELS = {
  ear_mean:         ['Apertura ocular',     'menor = tension'],
  ear_std:          ['Variab. ocular',      'inestabilidad de mirada'],
  blink_rate:       ['Parpadeos/s',         'normal ~0.25'],
  fear_rate:        ['Miedo facial',        'fraccion de frames'],
  neg_emotion_rate: ['Emociones negativas', 'fraccion de frames'],
  iris_ratio_mean:  ['Tamano iris',         'relativo al ojo'],
  iris_ratio_std:   ['Variab. iris',        'proxy de dilatacion'],
  ear_asymmetry:    ['Asimetria ocular',    'dif. ojo izq-der'],
  brow_asymmetry:   ['Asimetria cejas',     'dif. elevacion cejas'],
  gaze_std:         ['Inestab. mirada',     'movimiento del iris'],
  head_roll_std:    ['Giro lateral',        'inestabilidad'],
  head_pitch_std:   ['Cabeceo',             'inestabilidad'],
  head_yaw_std:     ['Giro horizontal',     'inestabilidad'],
  hr_bpm:           ['Frec. cardiaca',      'bpm estimado (rPPG)'],
  hr_std:           ['Variab. FC',          'proxy HRV'],
  pitch_mean:       ['Tono de voz',         'Hz medio'],
  pitch_std:        ['Variab. tono',        'jitter proxy'],
  energy_std:       ['Variab. energia',     'shimmer proxy'],
  pause_ratio:      ['Pausas en voz',       'fraccion silencio'],
  speech_rate:      ['Tasa de habla',       'fraccion activa'],
  filler_rate:      ['Muletillas/min',      'eh, um, bueno, pues...'],
  filler_count:     ['Total muletillas',    'detectadas por Whisper'],
  xgboost_bio:      ['XGBoost biométrico', 'solo canal facial+voz (0-1)'],
  contempt_proxy:   ['Desprecio (AU12)',    'asimetria de sonrisa — señal Ekman'],
  suppression_index:['Supresion facial',   'cara quieta = supresion activa'],
  emotion_variance: ['Varianza emocional',  'fluctuacion de expresion'],
  au_fear_peak:     ['Pico de miedo',       'max intensidad AU1+AU5'],
  // Deltas de calibración individual (Δ baseline vs análisis)
  ear_mean_delta:       ['Δ Apertura ocular',  '+ = mas cerrado en analisis'],
  ear_std_delta:        ['Δ Variab. ocular',   '+ = mas inestable'],
  ear_asymmetry_delta:  ['Δ Asimetria ocular', '+ = mas asimetrico'],
  brow_asymmetry_delta: ['Δ Asimetria cejas',  '+ = mas asimetrico'],
  gaze_std_delta:       ['Δ Mirada',           '+ = mas errante'],
  iris_ratio_mean_delta:['Δ Iris',             '+ = mas dilatado'],
  head_yaw_std_delta:   ['Δ Giro horiz.',      '+ = mas inestable'],
  head_pitch_std_delta: ['Δ Cabeceo',          '+ = mas inestable'],
  pitch_mean_delta:     ['Δ Tono voz',         '+ = tono mas alto'],
  pause_ratio_delta:    ['Δ Pausas',           '+ = mas pausas'],
  speech_rate_delta:    ['Δ Habla',            '- = habla mas lento'],
};

// ── Camara ────────────────────────────────────────────────────────────────────
async function startCamera() {
  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: true });
  } catch {
    stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    setStatus('warn', 'Microfono no disponible — solo analisis facial');
  }
  video.srcObject = stream;
  await video.play();
  return stream;
}

// ── Grabacion ─────────────────────────────────────────────────────────────────
async function startRecording() {
  const stream = video.srcObject || await startCamera();

  const mimeType = ['video/webm;codecs=vp8,opus', 'video/webm', 'video/mp4']
                    .find(t => MediaRecorder.isTypeSupported(t)) || '';

  chunks = [];
  _earBaseline.splice(0);   // reset baseline EAR al iniciar grabación
  lingBadge.classList.add('hidden');
  mediaRecorder = new MediaRecorder(stream, mimeType ? { mimeType } : {});
  mediaRecorder.ondataavailable = e => { if (e.data.size > 0) chunks.push(e.data); };
  mediaRecorder.onstop = sendVideo;
  mediaRecorder.start(200);

  btnRecord.querySelector('.hex-icon').textContent = '⏹';
  btnRecord.querySelector('.hex-label').textContent = 'DETENER';
  btnRecord.classList.add('recording');
  recDot.classList.remove('hidden');
  livePanel.classList.remove('hidden');
  btnDetails.classList.add('hidden');
  verdictCard.classList.add('hidden');
  analysisPanels.classList.add('hidden');

  secsLeft = MAX_SEC;
  countdown.classList.remove('hidden');
  countdown.textContent = secsLeft;
  setStatus('rec', `Grabando — habla con normalidad (${MAX_SEC}s max)`);
  _addLog(`Grabacion iniciada — ${MAX_SEC}s max`, 'REC');

  countdownId = setInterval(() => {
    secsLeft--;
    countdown.textContent = secsLeft;
    if (secsLeft <= 0) stopRecording();
  }, 1000);
}

function stopRecording() {
  if (!mediaRecorder || mediaRecorder.state === 'inactive') return;
  if (MAX_SEC - secsLeft < MIN_SEC) {
    setStatus('warn', `Graba al menos ${MIN_SEC} segundos`);
    return;
  }
  clearInterval(countdownId);
  recDot.classList.add('hidden');
  countdown.classList.add('hidden');
  livePanel.classList.add('hidden');
  mediaRecorder.stop();
  btnRecord.querySelector('.hex-icon').textContent = '⏺';
  btnRecord.querySelector('.hex-label').textContent = 'GRABAR';
  btnRecord.classList.remove('recording');
  btnRecord.disabled = true;
  setStatus('loading', 'Analizando — biometria + IA (puede tardar 20-40s)...');
  _addLog('Procesando — biometria + IA...', 'PROC');
}

// ── Envio ─────────────────────────────────────────────────────────────────────
async function sendVideo() {
  const blob = new Blob(chunks, { type: chunks[0]?.type || 'video/webm' });
  const form = new FormData();
  form.append('file', blob, 'recording.webm');

  try {
    const resp = await fetch('/analyze_video', { method: 'POST', body: form });
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.detail || `Error ${resp.status}`);
    showResults(data);
  } catch (err) {
    setStatus('warn', `Error: ${err.message}`);
  } finally {
    btnRecord.disabled = false;
  }
}

// ── Resultados ────────────────────────────────────────────────────────────────

function showResults({ lie_probability, features, transcription,
                       visual_analysis, linguistic_analysis, verdict: nemoVerdict }) {

  // ── Score integrado: PUNTUACION del LLM o fallback XGBoost ──────────────
  const _pMatch  = (nemoVerdict || '').match(/PUNTUACION:\s*(\d{1,3})/i);
  const _vMatch2 = (nemoVerdict || '').match(/VEREDICTO:\s*(AUTENTICO|AMBIGUO|SOSPECHOSO)/i);
  const _vKey    = _vMatch2?.[1]?.toUpperCase();

  const pct  = _pMatch ? Math.min(100, Math.max(0, parseInt(_pMatch[1])))
                       : Math.round(lie_probability * 100);
  const dash = Math.round(pct / 100 * RING_C);
  const col  = pct < 35 ? '#22c55e' : pct < 65 ? '#f59e0b' : '#ef4444';

  // Anillo
  ringFill.setAttribute('stroke-dasharray', `${dash} ${RING_C - dash}`);
  ringFill.style.stroke = col;
  scoreLabel.textContent = pct + '%';
  scoreLabel.style.color = col;

  // Texto veredicto
  verdict.textContent = _vKey === 'AUTENTICO'  ? 'Veracidad alta'
                      : _vKey === 'SOSPECHOSO' ? 'Posible engano detectado'
                      : _vKey === 'AMBIGUO'    ? 'Senales ambiguas'
                      : pct < 35 ? 'Veracidad alta'
                      : pct < 65 ? 'Senales ambiguas'
                                 : 'Posible engano detectado';
  verdict.style.color = col;

  // Score XGBoost puro → al grid de features como dato adicional
  if (features) features['xgboost_bio'] = Math.round(lie_probability * 100) / 100;

  // Linguistic risk badge
  const lingRisk = (linguistic_analysis || '').match(/RIESGO LINGUISTICO[^:]*:\s*(BAJO|MEDIO|ALTO)/i)?.[1];
  if (lingRisk) {
    const riskCol = RISK_COLORS[lingRisk] || '#94a3b8';
    lingBadge.textContent  = `Riesgo ling.: ${lingRisk}`;
    lingBadge.style.color  = riskCol;
    lingBadge.classList.remove('hidden');
  } else {
    lingBadge.classList.add('hidden');
  }

  // Veredicto — renderizar secciones estructuradas + markdown basico
  if (nemoVerdict) {
    verdictText.innerHTML = _renderStructured(nemoVerdict, {
      'CANAL BIOMETRICO':    '#7dd3fc',
      'CANAL LINGUISTICO':   '#7dd3fc',
      'CANAL VISUAL':        '#7dd3fc',
      'EVALUACION INTEGRADA':'#93c5fd',
      'VEREDICTO':           null,   // coloreado por valor
    });
    verdictCard.classList.remove('hidden');
  }

  // Paneles de analisis
  const hasAnalysis = transcription || visual_analysis || linguistic_analysis;
  if (hasAnalysis) {
    transcriptionText.textContent = transcription || '(no disponible)';
    visualText.textContent        = visual_analysis || '(no disponible)';
    linguisticText.innerHTML      = linguistic_analysis
      ? _renderStructured(linguistic_analysis, {
          'CBCA':                '#94a3b8',
          'REALITY MONITORING':  '#94a3b8',
          'ALERTAS':             '#fca5a5',
          'RIESGO LINGUISTICO':  null,   // coloreado por valor
        })
      : '(no disponible)';
    analysisPanels.classList.remove('hidden');
  }

  // Feature grid
  featureGrid.innerHTML = '';
  if (features) {
    for (const [key, val] of Object.entries(features)) {
      const [label, hint] = FEAT_LABELS[key] || [key, ''];
      const card = document.createElement('div');
      card.className = 'feat-card';
      card.innerHTML = `<span class="feat-name">${label}</span>
                        <span class="feat-val">${val}</span>
                        <span class="feat-hint">${hint}</span>`;
      featureGrid.appendChild(card);
    }
  }

  results.classList.remove('hidden');
  setStatus('ok', 'Analisis completado');

  // Populate kernel log for results screen
  _populateKernel({ lie_probability, features, transcription,
                    visual_analysis, linguistic_analysis, verdict: nemoVerdict });

  // Show "VER ANÁLISIS COMPLETO" button
  btnDetails.classList.remove('hidden');

  // Brief syslog summary
  const vLabel = _vKey === 'AUTENTICO'  ? 'AUTENTICO'
               : _vKey === 'SOSPECHOSO' ? 'SOSPECHOSO'
               : _vKey === 'AMBIGUO'    ? 'AMBIGUO'
               : pct < 35 ? 'AUTENTICO' : pct < 65 ? 'AMBIGUO' : 'SOSPECHOSO';
  _addLog(`Score integrado: ${pct}% — ${vLabel}`, 'DONE');
}

// ── Render texto estructurado con secciones coloreadas ────────────────────────
const RISK_COLORS = {
  BAJO:      '#22c55e',
  MEDIO:     '#f59e0b',
  ALTO:      '#ef4444',
  AUTENTICO: '#22c55e',
  AMBIGUO:   '#f59e0b',
  SOSPECHOSO:'#ef4444',
};

function _renderStructured(text, sectionColors) {
  const safe = text
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

  // Colorear valores clave (BAJO/MEDIO/ALTO/AUTENTICO/etc.)
  let html = safe.replace(
    /\b(BAJO|MEDIO|ALTO|AUTENTICO|AMBIGUO|SOSPECHOSO)\b/g,
    w => `<strong style="color:${RISK_COLORS[w] || '#f1f5f9'}">${w}</strong>`
  );

  // Resaltar cabeceras de sección (NOMBRE_SECCION: texto)
  for (const [section, color] of Object.entries(sectionColors || {})) {
    const re = new RegExp(`(${section}:)`, 'gi');
    const c  = color || '#f1f5f9';
    html = html.replace(re, `<strong style="color:${c}">$1</strong>`);
  }

  // Negrita markdown **texto**
  html = html.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');

  return html.replace(/\n/g, '<br>');
}

// ── Syslog helper ─────────────────────────────────────────────────────────────
let _logStart = Date.now();
function _addLog(text, key = '') {
  const cursor = sysLogBody.querySelector('.log-cursor');
  const line   = document.createElement('p');
  line.className = 'log-line';
  const safeText = text.replace(/</g, '&lt;').replace(/>/g, '&gt;');
  line.innerHTML = key
    ? `<span class="log-key">[${key}]</span> ${safeText}`
    : `&gt; ${safeText}`;
  if (cursor) sysLogBody.insertBefore(line, cursor);
  else sysLogBody.appendChild(line);
  sysLogBody.scrollTop = sysLogBody.scrollHeight;
}

// ── Kernel log populator ──────────────────────────────────────────────────────
function _populateKernel({ lie_probability, features, transcription,
                           visual_analysis, linguistic_analysis, verdict: nemo }) {
  const entries = [];
  let t = 0;
  const fmt = s => String(Math.floor(s/60)).padStart(2,'0') + ':' +
                   String(s % 60).padStart(2,'0') + ':00';

  entries.push([t++, 'INIT', 'Analysis pipeline complete — all channels processed']);
  entries.push([t++, 'BIO',  `XGBoost biometric raw score: ${Math.round(lie_probability * 100)}%`]);

  const pMatch = (nemo || '').match(/PUNTUACION:\s*(\d{1,3})/i);
  if (pMatch) entries.push([t++, 'SYNTH', `Integrated score (3-channel): ${pMatch[1]}%`]);

  const vMatch = (nemo || '').match(/VEREDICTO:\s*(AUTENTICO|AMBIGUO|SOSPECHOSO)/i);
  if (vMatch) entries.push([t++, 'VERDICT', vMatch[1]]);

  const lingRisk = (linguistic_analysis || '')
    .match(/RIESGO LINGUISTICO[^:]*:\s*(BAJO|MEDIO|ALTO)/i)?.[1];
  if (lingRisk) entries.push([t++, 'LING', `Linguistic risk level: ${lingRisk}`]);

  if (transcription) {
    const preview = transcription.length > 130
      ? transcription.slice(0, 130) + '...'
      : transcription;
    entries.push([t++, 'STT', `"${preview}"`]);
  }

  // Veredicto completo — líneas principales
  if (nemo) {
    const lines = nemo.split('\n').map(l => l.trim()).filter(l => l).slice(0, 14);
    for (const l of lines) entries.push([t++, '', l]);
  }

  kernelBody.innerHTML = '';
  const sanitize = s => s
    .replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')
    .replace(/\b(BAJO|MEDIO|ALTO|AUTENTICO|AMBIGUO|SOSPECHOSO)\b/g,
      w => `<strong style="color:${RISK_COLORS[w]||'var(--green)'}">${w}</strong>`);

  for (const [i, key, text] of entries) {
    const entry = document.createElement('div');
    entry.className = 'log-entry';
    const label = key
      ? `<strong style="color:var(--green)">[${key}]</strong> `
      : '';
    entry.innerHTML = `<span class="log-ts">${fmt(i)}</span><p>${label}${sanitize(text)}</p>`;
    kernelBody.appendChild(entry);
  }

  // Blinking cursor at end
  const cur = document.createElement('div');
  cur.className = 'log-entry';
  cur.innerHTML = `<span class="log-ts">--:--:--</span><p class="log-cursor">&gt; _</p>`;
  kernelBody.appendChild(cur);
}

// ── Screen transitions ────────────────────────────────────────────────────────
function _showScreen(el) {
  [screenLoading, screenMain, screenResults].forEach(s => s.classList.remove('active'));
  el.classList.add('active');
}

// ── Loading sequence ─────────────────────────────────────────────────────────
function _runLoader() {
  let pct = 0;
  const iv = setInterval(() => {
    pct += Math.random() * 1.8 + 0.4;
    if (pct >= 100) {
      pct = 100;
      progressBar.style.width = '100%';
      clearInterval(iv);
      setTimeout(() => {
        _showScreen(screenMain);
        _logStart = Date.now();
        startCamera()
          .then(() => { initFaceMesh(); })
          .catch(e => setStatus('warn', 'Error al acceder a la camara: ' + e.message));
      }, 350);
      return;
    }
    progressBar.style.width = pct + '%';
  }, 65); // ~4-5 s total
}

// ── Helpers ───────────────────────────────────────────────────────────────────
function setStatus(type, text) {
  statusText.textContent = text;
  statusBar.className = 'status-bar ' + type;
}

// ── Init ──────────────────────────────────────────────────────────────────────
btnRecord.addEventListener('click', () => {
  if (mediaRecorder && mediaRecorder.state === 'recording') stopRecording();
  else startRecording();
});

btnDetails.addEventListener('click', () => _showScreen(screenResults));
btnBack.addEventListener('click',    () => _showScreen(screenMain));

// ── Métricas en vivo (calculadas desde landmarks MediaPipe JS) ────────────────
const _LIVE_WIN  = 18;     // frames de suavizado
const _liveEars  = [];
const _liveGazeX = [];
const _liveGazeY = [];
const _liveHead  = [];
let   _prevNose  = null;

function _liveEAR(lm, idx) {
  const p = idx.map(i => lm[i]);
  const A = Math.hypot(p[1].x - p[2].x, p[1].y - p[2].y);
  const B = Math.hypot(p[4].x - p[5].x, p[4].y - p[5].y);
  const C = Math.hypot(p[0].x - p[3].x, p[0].y - p[3].y);
  return (A + B) / (2 * C + 1e-8);
}

function _std(arr) {
  if (arr.length < 2) return 0;
  const m = arr.reduce((a, b) => a + b, 0) / arr.length;
  return Math.sqrt(arr.reduce((s, v) => s + (v - m) ** 2, 0) / arr.length);
}

function _push(buf, val) {
  buf.push(val);
  if (buf.length > _LIVE_WIN) buf.shift();
  return buf.reduce((a, b) => a + b, 0) / buf.length;
}

function _setLiveBar(barEl, valEl, pct, label) {
  const p   = Math.min(100, Math.max(0, pct));
  const col = p < 35 ? '#22c55e' : p < 65 ? '#f59e0b' : '#ef4444';
  barEl.style.width           = `${Math.max(2, p)}%`;
  barEl.style.backgroundColor = col;
  valEl.textContent           = label;
  valEl.style.color           = col;
}

function updateLiveMetrics(lm) {
  // EAR: media ojo izquierdo + derecho
  const earL  = _liveEAR(lm, [33, 160, 158, 133, 153, 144]);
  const earR  = _liveEAR(lm, [362, 385, 387, 263, 373, 380]);
  const earM  = _push(_liveEars, (earL + earR) / 2);
  // Normalizar: 0.35 = muy abierto (0%), 0.18 = muy cerrado (100%)
  _setLiveBar(liveEarBar, liveEarVal,
    (0.35 - earM) / 0.17 * 100,
    earM.toFixed(2));

  // Gaze instability: std del centro de iris en ventana deslizante
  _liveGazeX.push((lm[468].x + lm[473].x) / 2);
  _liveGazeY.push((lm[468].y + lm[473].y) / 2);
  if (_liveGazeX.length > _LIVE_WIN) { _liveGazeX.shift(); _liveGazeY.shift(); }
  const gazeStd = Math.sqrt(_std(_liveGazeX) ** 2 + _std(_liveGazeY) ** 2);
  // Normalizar: 0 = estable (0%), 0.015+ = muy inestable (100%)
  _setLiveBar(liveGazeBar, liveGazeVal,
    gazeStd / 0.015 * 100,
    gazeStd.toFixed(3));

  // Head movement: desplazamiento de la punta de la nariz (landmark 4)
  const nx = lm[4].x, ny = lm[4].y;
  const move = _prevNose ? Math.hypot(nx - _prevNose[0], ny - _prevNose[1]) : 0;
  _prevNose  = [nx, ny];
  const headM = _push(_liveHead, move);
  // Normalizar: 0 = quieto (0%), 0.006+ = mucho movimiento (100%)
  _setLiveBar(liveHeadBar, liveHeadVal,
    headM / 0.006 * 100,
    (headM * 1000).toFixed(1));
}

// ── Malla facial MediaPipe JS ─────────────────────────────────────────────────
async function initFaceMesh() {
  try {
    const vision = await FilesetResolver.forVisionTasks(
      'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm'
    );
    const landmarker = await FaceLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
        delegate: 'GPU',
      },
      outputFaceBlendshapes: false,
      runningMode: 'VIDEO',
      numFaces: 1,
    });

    const ctx = meshCanvas.getContext('2d');

    // Buffer del canvas = resolución intrínseca del vídeo para que las
    // coordenadas normalizadas de MediaPipe mapeen directamente (sin DPR).
    // CSS width/height:100% se encarga de escalar al tamaño visual.
    function syncCanvas() {
      const vw = video.videoWidth  || video.clientWidth;
      const vh = video.videoHeight || video.clientHeight;
      if (meshCanvas.width !== vw || meshCanvas.height !== vh) {
        meshCanvas.width  = vw;
        meshCanvas.height = vh;
      }
    }

    let lastTs = -1;
    function render() {
      requestAnimationFrame(render);
      if (video.readyState < 2 || video.currentTime === lastTs) return;
      lastTs = video.currentTime;
      syncCanvas();

      const W = meshCanvas.width;
      const H = meshCanvas.height;
      const result = landmarker.detectForVideo(video, performance.now());
      ctx.clearRect(0, 0, W, H);

      if (!result.faceLandmarks?.length) {
        // Sin cara: vuelve suavemente a zoom neutro
        _autoZoom = _autoZoom * 0.98 + 1.0 * 0.02;
        _autoTx   = _autoTx   * 0.98;
        _autoTy   = _autoTy   * 0.98;
        _applyZoom();
        return;
      }

      const lm = result.faceLandmarks[0];

      // ── Auto-zoom: encuadra la cara en el óvalo ──────────────────────────
      let faceMinX = 1, faceMaxX = 0, faceMinY = 1, faceMaxY = 0;
      for (const p of lm) {
        if (p.x < faceMinX) faceMinX = p.x;
        if (p.x > faceMaxX) faceMaxX = p.x;
        if (p.y < faceMinY) faceMinY = p.y;
        if (p.y > faceMaxY) faceMaxY = p.y;
      }
      const faceH  = faceMaxY - faceMinY;
      const faceCX = (faceMinX + faceMaxX) / 2;
      const faceCY = (faceMinY + faceMaxY) / 2;

      // Objetivo: cara ocupa ~68% de la altura del óvalo (óvalo ≈ 0.45 del alto de vídeo)
      const TARGET_H   = 0.30;
      const targetZoom = Math.min(2.5, Math.max(1.0, TARGET_H / Math.max(faceH, 0.05)));
      const targetTx   = (0.5  - faceCX) * 100;
      const targetTy   = (0.49 - faceCY) * 100;

      // Interpolación suave para evitar saltos (α=0.04)
      _autoZoom = _autoZoom * 0.96 + targetZoom * 0.04;
      _autoTx   = _autoTx   * 0.96 + targetTx   * 0.04;
      _autoTy   = _autoTy   * 0.96 + targetTy   * 0.04;
      _applyZoom();

      // ── Color del mesh según calidad de encuadre ─────────────────────────
      const wellCentered = Math.abs(faceCX - 0.5) < 0.15 && Math.abs(faceCY - 0.49) < 0.15;
      const goodSize     = faceH > 0.18;
      const meshColor    = (wellCentered && goodSize)
        ? 'rgba(34, 211, 238, 0.70)'    // cyan: bien posicionado
        : 'rgba(245, 158, 11, 0.80)';   // ámbar: ajusta posición

      // ── Métricas en vivo ──────────────────────────────────────────────────
      const recording = mediaRecorder?.state === 'recording';
      if (recording) updateLiveMetrics(lm);

      // ── Dibuja landmarks con coloreado de anomalías ───────────────────────
      // Puntos normales (excluye ojos/iris en grabación para sobreescribir después)
      ctx.fillStyle = meshColor;
      for (let i = 0; i < lm.length; i++) {
        if (recording && (EYES_IDX.has(i) || IRIS_IDX.has(i))) continue;
        ctx.beginPath();
        ctx.arc(lm[i].x * W, lm[i].y * H, 1.6, 0, Math.PI * 2);
        ctx.fill();
      }

      if (recording) {
        // Anomalía EAR: cierre de ojos por debajo del baseline individual
        const earL   = _liveEAR(lm, [33,160,158,133,153,144]);
        const earR   = _liveEAR(lm, [362,385,387,263,373,380]);
        const earNow = (earL + earR) / 2;
        if (_earBaseline.length < 30) _earBaseline.push(earNow);
        const baseEAR    = _earBaseline.length > 0
          ? _earBaseline.reduce((a, b) => a + b) / _earBaseline.length : 0.30;
        const eyeAnomaly = earNow < baseEAR * 0.86;   // 14% caída — tensión o parpadeo

        // Anomalía gaze: picos en std de iris en ventana reciente
        const gazeAnomaly = _liveGazeX.length >= 8 &&
          Math.sqrt(_std(_liveGazeX.slice(-8)) ** 2 + _std(_liveGazeY.slice(-8)) ** 2) > 0.009;

        // Ojos
        ctx.fillStyle = eyeAnomaly ? 'rgba(239, 68, 68, 0.90)' : meshColor;
        for (const i of EYES_IDX) {
          ctx.beginPath();
          ctx.arc(lm[i].x * W, lm[i].y * H, 2.0, 0, Math.PI * 2);
          ctx.fill();
        }
        // Iris
        ctx.fillStyle = gazeAnomaly ? 'rgba(251, 146, 60, 0.95)' : meshColor;
        for (const i of IRIS_IDX) {
          ctx.beginPath();
          ctx.arc(lm[i].x * W, lm[i].y * H, 2.2, 0, Math.PI * 2);
          ctx.fill();
        }
      } else {
        // Fuera de grabación: ojos e iris con color normal
        ctx.fillStyle = meshColor;
        for (const i of EYES_IDX) {
          ctx.beginPath();
          ctx.arc(lm[i].x * W, lm[i].y * H, 1.6, 0, Math.PI * 2);
          ctx.fill();
        }
        for (const i of IRIS_IDX) {
          ctx.beginPath();
          ctx.arc(lm[i].x * W, lm[i].y * H, 1.6, 0, Math.PI * 2);
          ctx.fill();
        }
      }
    }

    meshCanvas.classList.add('ready');
    render();
  } catch (e) {
    // Si falla (sin conexion, navegador incompatible) la app sigue funcionando
    console.warn('[FaceMesh]', e);
  }
}

// ── Zoom ──────────────────────────────────────────────────────────────────────
const videoInner = document.getElementById('videoInner');
function _applyZoom() {
  const z = Math.min(2.5, Math.max(1.0, _autoZoom * _zoomMul));
  // scale() desde el centro del elemento (transform-origin: 50% 50% por defecto)
  // translate() mueve la cara al centro del óvalo antes de escalar
  videoInner.style.transform = `scale(${z.toFixed(3)}) translate(${_autoTx.toFixed(2)}%, ${_autoTy.toFixed(2)}%)`;
}
// +/- ajustan el multiplicador manual sobre el auto-zoom
document.getElementById('btnZoomIn') .addEventListener('click', () => { _zoomMul = Math.min(2.5, _zoomMul + 0.15); _applyZoom(); });
document.getElementById('btnZoomOut').addEventListener('click', () => { _zoomMul = Math.max(0.4, _zoomMul - 0.15); _applyZoom(); });

_runLoader();
