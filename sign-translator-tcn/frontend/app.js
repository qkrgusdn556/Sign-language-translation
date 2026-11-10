import { STORAGE_KEY, MAX_SAMPLES, validateSamples, landmarksToVector, classify, classifySequence, staticComparisonSamples, trackedHandCount, motionMagnitude, createStabilizer, assignHandSlots, createShoulderTracker, createArmTracker, poseArmTargets } from "./recognition.js?v=arm-hand-distance-1";
import { loadYolo11Classifier, inferYolo11, createStillnessGate, drawYoloBoxes } from "./yolo.js?v=yolo26-1";

import { createTcnStream } from "./tcn_stream.js?v=per-hand-motion-1";
import { createFingerspelling, createAutomaticFingerspelling } from "./fingerspell.js?v=yolo-only-1";
import { createInputActivity } from "./input_activity.js?v=jamo-small-motion-1";
import { createHandTracker } from "./hand_tracking.js?v=palm-dedupe-1";
import {createHandPrecision} from "./hand_precision.js?v=precision-1";
import {compareFingerspell, fingerspellAgreement, isJamo} from "./fingerspell_consensus.js?v=settled-pose-1";
import {installCapturePanel} from "./raw_capture.js?v=collection-evaluation-1";
import {installSharedMotion} from "./shared_motion.js?v=shared-motion-1";
import {REST_LABEL,isRestLabel,createRestPoseGate} from "./rest_pose.js?v=rest-dtw-4";

let rawCapture=null,sharedMotion=null;
const fingerprint=async bytes=>Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',bytes)),byte=>byte.toString(16).padStart(2,'0')).join('');

const inputActivity = createInputActivity();
const fingerspellActivity = createInputActivity({profile:'fingerspell'});
const restGate=createRestPoseGate();
const shoulderTracker = createShoulderTracker();
const armTracker=createArmTracker();
const handTracker = createHandTracker();
const motionTracker = createHandTracker();
let trackingEpoch=0;
const precisionToggle=document.querySelector('#precisionTracking');
const handPrecision=createHandPrecision({onStatus:message=>{document.querySelector('#precisionLabel').textContent=message;}});
precisionToggle.addEventListener('change',()=>{
  trackingEpoch++;handTracker.reset();motionTracker.reset();resetInputBoundary();
  if(precisionToggle.checked&&state.running)handPrecision.start();
  else {handPrecision.stop();document.querySelector('#precisionLabel').textContent='손 정밀 추적';}
});
let fingerspellFrames=[], fingerspellFrameAt=-Infinity, fingerspellSamplesVersion=null;

function resetFingerspellFrames() {
  fingerspellFrames=[];fingerspellFrameAt=-Infinity;
}

function observeFingerspellFrame(vector,now) {
  if(!vector) { resetFingerspellFrames();return; }
  if(now-fingerspellFrameAt>250 || fingerspellSamplesVersion!==state.samples) resetFingerspellFrames();
  fingerspellSamplesVersion=state.samples;
  if(now-fingerspellFrameAt<75) return;
  fingerspellFrameAt=now;
  fingerspellFrames.push([...vector]);
  if(fingerspellFrames.length>20) fingerspellFrames.shift();
}

const wordSamples = () => state.samples.filter(row=>!isJamo(row.label)&&!isRestLabel(row.label));
const sampleName=label=>isRestLabel(label)?'수어 대기 자세':label;
const jointOnly = () => document.querySelector('#fingerspellMethod')?.value==='joints';
const yoloOnly = () => document.querySelector('#fingerspellMethod')?.value==='yolo';
const spellingMethod = () => jointOnly()?'joints':yoloOnly()?'yolo':'consensus';

let fingerspelling = false, fingerspellEpoch = 0;
const protectedWordIndices = new Set();
const spelling = createFingerspelling(letters => window.Hangul.assemble(letters));
let automaticWordIndex = null;
const automaticSpelling = createAutomaticFingerspelling(letters => window.Hangul.assemble(letters), {
  onText(text) {
    if (automaticWordIndex === null) {
      automaticWordIndex = state.sentenceWords.length;
      protectedWordIndices.add(automaticWordIndex);
      state.sentenceWords.push(text);
    } else state.sentenceWords[automaticWordIndex] = text;
    rawCapture?.prediction(text,"jamo",`auto-${automaticWordIndex}`);
    state.polishedSentence = formatSentence(state.sentenceWords, state.speechStyle);
    renderSentence();
    renderSpelling();
  },
  onEnd() { automaticWordIndex = null; renderSpelling(); },
});

function automaticEnabled() {
  return !fingerspelling && document.querySelector("#autoFingerspell").checked &&
    (jointOnly()||Boolean(state.yolo)) && !state.yoloLoading && !state.recording;
}

function allowWordCommit(label) {
  if(state.restActive||isRestLabel(label))return false;
  if (!automaticEnabled()) return true;
  if (automaticSpelling.blocksWord(performance.now())) return false;
  automaticSpelling.finish();
  return true;
}

function renderSpelling() {
  document.querySelector("#fingerspellText").textContent = fingerspelling ? spelling.text() : automaticSpelling.text();
  for (const id of ["fingerspellFinish", "fingerspellBackspace", "fingerspellRepeat", "fingerspellCancel"]) {
    document.querySelector(`#${id}`).disabled = !fingerspelling || Boolean(rawCapture?.busy()||sharedMotion?.busy());
  }
  document.querySelector("#fingerspellStart").disabled = fingerspelling || Boolean(rawCapture?.busy()||sharedMotion?.busy());
  modeSelect.disabled = fingerspelling || Boolean(rawCapture?.busy()||sharedMotion?.busy());
  trainTcnBtn.disabled = fingerspelling || Boolean(rawCapture?.busy()||sharedMotion?.busy());
  document.querySelector("#fingerspellStatus").textContent = fingerspelling ? "수동 지문자 입력 중"
    : automaticEnabled() ? automaticSpelling.text() ? "자동 지문자 조합 중" : "자동 지문자 대기"
    : "일반 수어 모드";
  if (!fingerspelling && document.querySelector('#autoFingerspell').checked) {
    const missing = [];
    if (!yoloOnly() && state.storageBlocked) missing.push('저장 샘플 연결 실패');
    else if (!yoloOnly() && !state.samples.some(row => isJamo(row.label))) missing.push('지문자 관절 샘플 없음');
    if (!jointOnly() && !state.yolo) missing.push(state.yoloLoading ? 'YOLO 모델 로딩 중' : 'YOLO 모델 미연결');
    else if (!jointOnly() && !state.yolo.labels.some(isJamo)) missing.push('YOLO 지문자 라벨 없음');
    if (missing.length) document.querySelector('#fingerspellStatus').textContent = `자동 지문자 입력 불가 · ${missing.join(' · ')}`;
  }
}

function resetInputBoundary() {
  restGate.reset();state.restActive=false;
  state.fingerspellPreview=null;state.fingerspellStill=false;
  resetFingerspellFrames();
  inputActivity.reset();
  fingerspellActivity.reset();
  fingerspellEpoch++;
  tcnStream.reset();
  state.stabilizer.reset();
  state.motion = { lastVector: null, lastAt: 0, active: false, quietSince: 0, frames: [] };
  state.stillness.reset();
  state.lastWrists = null;
  state.sentenceArmed = true;
}

function startSpelling() {
  if ((!jointOnly()&&!state.yolo) || state.recording) { notify("YOLO 단독·교차 확인은 ONNX 모델 연결이 필요합니다.", true); return; }
  if (!jointOnly()&&!state.yolo.labels.some(label => /^[ㄱ-ㅎㅏ-ㅣ]$/u.test(label))) { notify("자음·모음 라벨을 가진 지문자 모델이 필요합니다.", true); return; }
  automaticSpelling.finish();
  fingerspelling = true;
  spelling.clear();
  resetInputBoundary();
  renderSpelling();
  updateControls();
}

function endSpelling(commit = true) {
  if (!fingerspelling) return;
  const text = spelling.text();
  if (commit && text) {
    rawCapture?.prediction(text,"jamo");
    protectedWordIndices.add(state.sentenceWords.length);
    state.sentenceWords.push(text);
    state.polishedSentence = formatSentence(state.sentenceWords, state.speechStyle);
    renderSentence();
  }
  fingerspelling = false;
  spelling.clear();
  resetInputBoundary();
  renderSpelling();
  updateControls();
}

document.querySelector("#fingerspellStart").addEventListener("click", startSpelling);
document.querySelector("#fingerspellFinish").addEventListener("click", () => endSpelling());
document.querySelector("#fingerspellCancel").addEventListener("click", () => endSpelling(false));
document.querySelector("#fingerspellBackspace").addEventListener("click", () => { spelling.backspace(); renderSpelling(); });
document.querySelector("#fingerspellRepeat").addEventListener("click", () => { spelling.release(); renderSpelling(); });
document.querySelector("#autoFingerspell").addEventListener("change", () => {
  automaticSpelling.finish();
  resetInputBoundary();
  renderSpelling();
});
document.querySelector('#fingerspellMethod').addEventListener('change',()=>{
  automaticSpelling.finish();endSpelling(false);resetInputBoundary();renderSpelling();
});

let tcnEnabled = false, tcnGeneration = 0;
const modeSelect = document.querySelector("#recognizerMode");
const tcnStatus = document.querySelector("#tcnStatus");
const trainTcnBtn = document.querySelector("#trainTcnBtn");
const tcnStream = createTcnStream({
  async infer(sequence) {
    const response = await fetch("/api/tcn/predict", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ sequence }), signal: AbortSignal.timeout(2500) });
    if (!response.ok) throw new Error("TCN 연결 실패");
    return response.json();
  },
  onResult: handleTcnResult,
  onError() {
    tcnEnabled = false;
    modeSelect.value = "dtw";
    tcnStream.reset();
    tcnStatus.textContent = "TCN 연결 실패 · DTW 비교 모드";
  },
});

function handleTcnResult(result) {
    if (!tcnEnabled || !state.running || state.recording || fingerspelling) return;
    if(state.restActive||isRestLabel(result.label))return;
    if(isJamo(result.label)) {
      showPrediction(`지문자 후보 ${result.label}`, "TCN 임시 후보 · DTW 관절 비교와 YOLO11의 일치를 기다립니다.",result.similarity||0,result.label,"TCN",true);
      return;
    }
    const reason = { "inactive-input": "손 이동 확인 대기 · 아직 모델에 보내지 않음", "motion-settled": "손 움직임 종료 · 새 단어 확정 보류", "tracking-missing": "관절 추적 복귀 대기 · 이전 후보 해제", "insufficient-motion": "입력 구간의 동작 근거 부족", "low-score": "후보 점수 85% 미만",
      "small-margin": "1·2위 차이 30%포인트 미만", "duplicate": "이미 출력한 단어", "confirmed": "단어 확정" }[result.reason]
      || `후보 확인 ${result.votes || 0}/3회 · ${Math.round(result.heldMs || 0)}/300ms`;
    showPrediction(result.status === "match" ? result.label : result.label ? "확인 중" : "동작 대기", `TCN · ${reason} · 점수는 정확도가 아닙니다`, result.similarity, result.label, "TCN", result.status !== "match");
    if (result.commit) {
      if (!allowWordCommit(result.label)) { tcnStream.rejectCommit(result.label); return; }
      state.sentenceWords.push(result.label);
      rawCapture?.prediction(result.label,"tcn");
      state.polishedSentence = formatSentence(state.sentenceWords, state.speechStyle);
      renderSentence();
    }
}

async function connectTcn() {
  automaticSpelling.finish();
  const generation = ++tcnGeneration;
  tcnEnabled = false;
  tcnStream.reset();
  state.stabilizer.reset();
  state.motion = { lastVector: null, lastAt: 0, active: false, quietSince: 0, frames: [] };
  if (modeSelect.value !== "tcn") { tcnStatus.textContent = "DTW 비교 모드"; return; }
  tcnStatus.textContent = "TCN 연결 중";
  try {
    const response = await fetch("/api/tcn/status", { signal: AbortSignal.timeout(15000) });
    const info = await response.json();
    if (generation !== tcnGeneration) return;
    if (!response.ok || !info.ready || info.input_dim !== 130 || info.feature_schema !== "shoulder-hands-130-v1") throw new Error();
    tcnEnabled = true;
    tcnStatus.textContent = `TCN 실험 · ${info.labels.join(" / ")} · 학습 ${info.original_recordings ?? "?"}개 · 연결 동작 미학습${info.sliding_window_augmented ? " · 최근 구간 학습됨" : " · 이전 모델: 최근 구간 재학습 권장"}${info.excluded_labels?.length ? ` · 재수집 필요: ${info.excluded_labels.join(" / ")}` : ""}`;
    if (!state.running) showPrediction("대기 중", "한국 수어 TCN 시험 모델 연결됨", 0);
  } catch {
    if (generation !== tcnGeneration) return;
    modeSelect.value = "dtw";
    tcnStatus.textContent = "어깨 기준 TCN 학습 필요 · DTW 비교 모드";
  }
}
modeSelect?.addEventListener("change", connectTcn);
trainTcnBtn?.addEventListener("click", async () => {
  if (state.saving || state.storageLoading || state.storageBlocked) { notify("저장 상태를 확인해 주세요.", true); return; }
  stopCamera();
  trainTcnBtn.disabled = true;
  try {
    const response = await fetch("/api/tcn/train", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ train: true }), signal: AbortSignal.timeout(15000) });
    const info = await response.json();
    if (!response.ok) throw new Error(info.error || "학습 시작 실패");
    let completed = false;
    for (let attempt = 0; attempt < 600; attempt++) {
      tcnStatus.textContent = `TCN 학습 중 · 사용 ${info.recordings}개 · 제외 ${info.excluded_recordings}개`;
      await new Promise(resolve => setTimeout(resolve, 1000));
      const statusResponse = await fetch("/api/tcn/status", { signal: AbortSignal.timeout(15000) });
      if (!statusResponse.ok) throw new Error("학습 상태 확인 실패");
      const status = await statusResponse.json();
      if (status.job === "failed") throw new Error(status.error || "학습 실패");
      if (status.job === "done") { completed = true; break; }
    }
    if (!completed) throw new Error("학습 상태 확인 시간이 초과되었습니다. 서버 상태를 확인해 주세요.");
    await connectTcn();
    notify("TCN 학습 완료 · 새 촬영으로 검증해 주세요.");
  } catch (error) { tcnStatus.textContent = `TCN 학습 확인 실패 · ${error.message}`; notify(error.message, true); }
  finally { trainTcnBtn.disabled = false; }
});


let HAND_CONNECTIONS = [];
let icons;

const els = {
  video: document.querySelector("#camera"),
  canvas: document.querySelector("#overlay"),
  prediction: document.querySelector("#prediction"),
  confidence: document.querySelector("#confidence"),
  confidenceBar: document.querySelector("#confidenceBar"),
  labelInput: document.querySelector("#labelInput"),
  startBtn: document.querySelector("#startBtn"),
  sampleBtn: document.querySelector("#sampleBtn"),
  recordBtn: document.querySelector("#recordBtn"),
  clearBtn: document.querySelector("#clearBtn"),
  sampleCount: document.querySelector("#sampleCount"),
  trackingState: document.querySelector("#trackingState"),
  labels: document.querySelector("#labels"),
  cameraEmpty: document.querySelector("#cameraEmpty"),
  notice: document.querySelector("#notice"),
  noticeText: document.querySelector("#noticeText"),
  undoBtn: document.querySelector("#undoBtn"),
  exportBtn: document.querySelector("#exportBtn"),
  importBtn: document.querySelector("#importBtn"),
  importFile: document.querySelector("#importFile"),
  clearDialog: document.querySelector("#clearDialog"),
  clearConfirm: document.querySelector("#clearConfirm"),
  captureProgress: document.querySelector("#captureProgress"),
  similarity: document.querySelector("#similarity"),
  nearestSample: document.querySelector("#nearestSample"),
  overlayToggle: document.querySelector("#overlayToggle"),
  faceState: document.querySelector("#faceState"),
  yoloModelFile: document.querySelector("#yoloModelFile"),
  yoloLabelsFile: document.querySelector("#yoloLabelsFile"),
  yoloLoadBtn: document.querySelector("#yoloLoadBtn"),
  yoloUnloadBtn: document.querySelector("#yoloUnloadBtn"),
  yoloState: document.querySelector("#yoloState"),
  yoloResult: document.querySelector("#yoloResult"),
  sentenceText: document.querySelector("#sentenceText"),
  clearSentenceBtn: document.querySelector("#clearSentenceBtn"),
  polishSentenceBtn: document.querySelector("#polishSentenceBtn"),
  polishedSentenceText: document.querySelector("#polishedSentenceText"),
  politeStyle: document.querySelector("#politeStyle"),
  casualStyle: document.querySelector("#casualStyle"),
};

const state = {
  landmarker: null,
  poseLandmarker: null,
  shoulders: null,
  shoulderHeld: false,
  restActive: false,
  fingerspellVector: null,
  fingerspellStill: false,
  fingerspellPreview: null,
  pose: null,
  poseArms: null,
  running: false,
  starting: false,
  importing: false,
  stream: null,
  frameId: 0,
  lastVideoTime: -1,
  lastFrameAt: 0,
  lastPoseAt: 0,
  lastVector: null,
  slots: [null, null],
  samples: [],
  recording: null,
  undo: null,
  storageBlocked: false,
  storageLoading: typeof fetch !== "undefined" || typeof indexedDB !== "undefined",
  fileStorage: false,
  saving: false,
  database: null,
  stabilizer: createStabilizer(),
  yolo: null,
  yoloLoading: false,
  yoloBusy: false,
  yoloLastAt: 0,
  stillness: createStillnessGate(400, 0.015),
  lastWrists: null,
  sentenceWords: [],
  sentenceArmed: true,
  speechStyle: "polite",
  polishedSentence: "",
  motion: { lastVector: null, lastAt: 0, active: false, quietSince: 0, frames: [] },
  partialMatchAllowed: false,
};

els.startBtn.addEventListener("click", () => state.running ? stopCamera() : startCamera());
els.sampleBtn.addEventListener("click", () => addSample(els.labelInput.value));
els.recordBtn.addEventListener("click", recordSequence);
document.querySelector('#restPoseSave').addEventListener('click',()=>{resetInputBoundary();automaticSpelling.finish();addSample(REST_LABEL);});
document.querySelector('#restPoseRecord').addEventListener('click',()=>recordSequence({rest:true}));
document.querySelector('#restGuardToggle').addEventListener('change',resetInputBoundary);
els.clearBtn.addEventListener("click", () => els.clearDialog.showModal());
els.clearConfirm.addEventListener("click", clearSamples);
document.querySelector("#clearCancel").addEventListener("click", () => els.clearDialog.close());
els.labelInput.addEventListener("input", updateControls);
els.exportBtn.addEventListener("click", exportSamples);
els.importBtn.addEventListener("click", () => els.importFile.click());
els.importFile.addEventListener("change", importSamples);
els.undoBtn.addEventListener("click", undoSamples);
els.overlayToggle.addEventListener("change", () => { els.canvas.hidden = !els.overlayToggle.checked; });
els.yoloLoadBtn.addEventListener("click", loadYolo);
els.yoloUnloadBtn.addEventListener("click", unloadYolo);
els.clearSentenceBtn.addEventListener("click", clearSentence);
els.polishSentenceBtn.addEventListener("click", polishCurrentSentence);
els.politeStyle.addEventListener("change", () => setSpeechStyle("polite"));
els.casualStyle.addEventListener("change", () => setSpeechStyle("casual"));
window.addEventListener("pagehide", () => stopCamera());
document.addEventListener("visibilitychange", () => {
  if (document.hidden && (state.running || state.recording)) stopCamera("화면을 벗어나 카메라를 중지했습니다.");
});

state.samples = loadSamples();
updateSampleUi();
renderSentence();
if (typeof fetch !== "undefined") await initializeFileStorage();
else if (state.storageLoading) await initializeSampleStorage();
loadIcons();
if(typeof document.getElementById==='function'&&document.getElementById('rawCaptureStart')) {
  rawCapture=installCapturePanel({getStream:()=>state.stream,getPipeline:()=>`${tcnEnabled?'tcn':'dtw'}|jamo-${spellingMethod()}`,
    getModelIdentity:async()=>({samples_sha256:await fingerprint(new TextEncoder().encode(JSON.stringify(state.samples))),yolo:state.yoloFingerprint||null,precision_tracking:precisionToggle.checked}),
    onSaved:()=>sharedMotion?.refresh(),
    onBegin:()=>{if(state.recording||sharedMotion?.busy()||fingerspelling)throw Error('기존 동작 수집·공동 저장·수동 지문자 입력을 먼저 종료해 주세요.');automaticSpelling.finish();resetInputBoundary();},
    onChange:()=>{
      renderSpelling();updateYoloUi();document.querySelector('#autoFingerspell').disabled=false;
      document.querySelector('#fingerspellMethod').disabled=false;
      trainTcnBtn.disabled=fingerspelling;updateControls();
    }});
  sharedMotion=installSharedMotion({getStream:()=>state.stream,getVideo:()=>els.video,getPipeline:()=>`${tcnEnabled?'tcn':'dtw'}|jamo-${spellingMethod()}`,
    getModelIdentity:async()=>({samples_sha256:await fingerprint(new TextEncoder().encode(JSON.stringify(state.samples))),yolo:state.yoloFingerprint||null,precision_tracking:precisionToggle.checked}),
    onChange:()=>{rawCapture?.setExternalBusy(sharedMotion?.busy());renderSpelling();updateYoloUi();for(const id of ['autoFingerspell','fingerspellMethod'])document.querySelector(`#${id}`).disabled=false;updateControls();},
    onCommitted:async response=>{await initializeFileStorage();if(state.storageBlocked)throw Error('저장소 연결을 확인하고 새로고침해 주세요.');state.undo=null;rawCapture?.noteSharedSession(response.clip.session_id);},beforeTraining:()=>stopCamera()});
}
if (typeof fetch !== "undefined") await connectTcn();

async function loadIcons() {
  try {
    icons = await import("https://cdn.jsdelivr.net/npm/lucide@0.468.0/+esm");
    for (const [id, name] of [["exportBtn", "Download"], ["importBtn", "Upload"], ["undoBtn", "Undo2"], ["clearBtn", "Trash2"]]) {
      els[id].prepend(icons.createElement(icons[name], { width: 16, height: 16, "aria-hidden": "true" }));
    }
    document.querySelector(".camera-symbol").replaceChildren(icons.createElement(icons.Camera, { width: 32, height: 32, "aria-hidden": "true" }));
    updateSampleUi();
  } catch { /* Text controls remain available if the optional icon CDN is offline. */ }
}

async function startCamera() {
  if (state.starting || state.running) return;
  state.starting = true;
  updateControls();
  els.startBtn.textContent = "준비 중";
  els.trackingState.textContent = "모델 로딩";
  try {
    if (!navigator.mediaDevices?.getUserMedia) throw new Error("카메라를 사용할 수 없는 브라우저입니다.");
    if (!state.landmarker || !state.poseLandmarker) {
      const { HandLandmarker, PoseLandmarker, FilesetResolver } = await import("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.22-rc.20250304");
      HAND_CONNECTIONS = HandLandmarker.HAND_CONNECTIONS;
      const vision = await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.22-rc.20250304/wasm");
      const options = {
        baseOptions: {
          modelAssetPath: "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
          delegate: "GPU",
        },
        runningMode: "VIDEO", numHands: 2,
        minHandDetectionConfidence: 0.6, minHandPresenceConfidence: 0.6, minTrackingConfidence: 0.6,
      };
      if (!state.landmarker) {
        try {
          state.landmarker = await HandLandmarker.createFromOptions(vision, options);
        } catch {
          options.baseOptions.delegate = "CPU";
          state.landmarker = await HandLandmarker.createFromOptions(vision, options);
        }
      }
      if (!state.poseLandmarker) {
        els.trackingState.textContent = "팔 모델 로딩";
        const poseOptions = {
          baseOptions: {
            modelAssetPath: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task",
            delegate: "GPU",
          },
          runningMode: "VIDEO", numPoses: 1,
          minPoseDetectionConfidence: 0.35, minPosePresenceConfidence: 0.35, minTrackingConfidence: 0.35,
        };
        try {
          state.poseLandmarker = await PoseLandmarker.createFromOptions(vision, poseOptions);
        } catch {
          poseOptions.baseOptions.delegate = "CPU";
          state.poseLandmarker = await PoseLandmarker.createFromOptions(vision, poseOptions);
        }
      }
    }
    els.trackingState.textContent = "카메라 연결";
    state.stream = await navigator.mediaDevices.getUserMedia({ video: {
      width: { ideal: 960 }, height: { ideal: 540 }, facingMode: "user", frameRate: { ideal: 30, max: 30 },
    }, audio: false });
    if (document.hidden) { stopCamera(); return; }
    els.video.srcObject = state.stream;
    await els.video.play();
    if (document.hidden) { stopCamera(); return; }
    state.stream.getVideoTracks().forEach((track) => track.addEventListener("ended", () => stopCamera("카메라 연결이 끊어졌습니다."), { once: true }));
    state.running = true;
    if(precisionToggle.checked)handPrecision.start();
    state.lastVideoTime = -1;
    state.lastPoseAt = 0;
    els.cameraEmpty.hidden = true;
    els.trackingState.textContent = "손 대기";
    state.frameId = requestAnimationFrame(loop);
  } catch (error) {
    stopCamera();
    const messages = {
      NotAllowedError: "카메라 권한이 차단되었습니다. 주소창의 카메라 권한을 허용한 뒤 다시 시작해 주세요.",
      NotFoundError: "연결된 카메라를 찾지 못했습니다.",
      NotReadableError: "카메라를 열지 못했습니다. 다른 앱에서 사용 중인지 확인해 주세요.",
    };
    notify(messages[error.name] || "카메라 준비에 실패했습니다. 네트워크와 카메라 연결을 확인한 뒤 다시 시도해 주세요.", true);
  } finally {
    state.starting = false;
    updateControls();
  }
}

async function loadYolo() {
  state.yoloBoxes=null;
  if (state.yoloLoading) return;
  automaticSpelling.finish();
  fingerspellEpoch++;
  state.yoloLoading = true;
  updateYoloUi("모델 로딩 중");
  try {
    state.yolo = await loadYolo11Classifier(els.yoloModelFile.files[0], els.yoloLabelsFile.files[0]);
    state.yoloFingerprint={model_sha256:await fingerprint(await els.yoloModelFile.files[0].arrayBuffer()),labels_sha256:await fingerprint(new TextEncoder().encode(JSON.stringify(state.yolo.labels)))};
    state.stillness.reset();
    state.lastWrists = null;
    els.yoloResult.textContent = `${state.yolo.modelName} · ${state.yolo.labels.length}개 라벨 연결됨 · ${state.yolo.task==='detect'?'탐지 박스 표시 준비':'사진 분류 모델 · 위치 박스 없음'}`;
    updateYoloUi("연결됨");
  } catch (error) {
    state.yolo = null;
    state.yoloFingerprint=null;
    updateYoloUi("연결 실패");
    els.yoloResult.textContent = error.message || "YOLO11 모델을 불러오지 못했습니다.";
  } finally {
    state.yoloLoading = false;
    updateYoloUi();
    renderSpelling();
  }
}

function unloadYolo() {
  state.yoloBoxes=null;
  automaticSpelling.finish();
  fingerspellEpoch++;
  endSpelling(false);
  state.yolo = null;
  state.yoloFingerprint=null;
  state.yoloBusy = false;
  state.stillness.reset();
  state.lastWrists = null;
  state.motion = { lastVector: null, lastAt: 0, active: false, quietSince: 0, frames: [] };
  state.motion = { lastVector: null, lastAt: 0, active: false, quietSince: 0, frames: [] };
  els.yoloModelFile.value = "";
  els.yoloLabelsFile.value = "";
  els.yoloResult.textContent = "YOLO11 연결을 해제했습니다.";
  renderSpelling();
  updateYoloUi("모델 없음");
}

function updateYoloUi(status) {
  els.yoloState.textContent = status || (state.yolo ? `${state.yolo.modelName || 'YOLO'} 연결됨` : "모델 없음");
  els.yoloLoadBtn.disabled = state.yoloLoading;
  els.yoloUnloadBtn.disabled = !state.yolo || state.yoloLoading;
  els.yoloModelFile.disabled = state.yoloLoading;
  els.yoloLabelsFile.disabled = state.yoloLoading;
}

function stopCamera(message) {
  state.yoloBoxes=null;
  restGate.reset();state.restActive=false;
  sharedMotion?.cancel();
  trackingEpoch++;handPrecision.stop();
  document.querySelector('#precisionLabel').textContent='손 정밀 추적';
  rawCapture?.stop();
  state.fingerspellStill=false;state.fingerspellPreview=null;
  state.fingerspellVector=null;
  resetFingerspellFrames();
  handTracker.reset();
  motionTracker.reset();
  shoulderTracker.reset();
  armTracker.reset();
  state.shoulderHeld = false;
  automaticSpelling.finish();
  automaticSpelling.pause({lost:true});
  inputActivity.reset();
  fingerspellActivity.reset();
  fingerspellEpoch++;
  spelling.resetCandidate();
  tcnStream.reset();
  state.partialMatchAllowed = false;
  state.running = false;
  state.motion = { lastVector: null, lastAt: 0, active: false, quietSince: 0, frames: [] };
  cancelAnimationFrame(state.frameId);
  state.stream?.getTracks().forEach((track) => track.stop());
  state.stream = null;
  els.video.pause();
  els.video.srcObject = null;
  state.lastVector = null;
  state.shoulders = null;
  state.lastPoseAt = 0;
  state.pose = null;
  state.poseArms = null;
  state.stillness.reset();
  state.lastWrists = null;
  els.faceState.textContent = "어깨 대기";
  state.slots = [null, null];
  state.stabilizer.reset();
  finishRecording(false);
  els.canvas.getContext("2d").clearRect(0, 0, els.canvas.width, els.canvas.height);
  els.cameraEmpty.hidden = false;
  els.trackingState.textContent = "중지";
  showPrediction("대기 중", "저장된 손모양과 비교할 준비가 되었습니다.", 0);
  updateControls();
  if (message) notify(message);
}

async function loop(now) {
  if (!state.running) return;
  automaticSpelling.tick(now);
  try {
    if (els.video.readyState >= 2 && els.video.currentTime !== state.lastVideoTime && now - state.lastFrameAt >= 45) {
      state.lastVideoTime = els.video.currentTime;
      state.lastFrameAt = now;
      resizeCanvas();
      const epoch=trackingEpoch,stream=state.stream,mediaTime=els.video.currentTime*1000,started=performance.now();
      const rawHands=state.landmarker.detectForVideo(els.video,now);
      const motionHands=motionTracker.update(rawHands,now);
      let detectedHands=rawHands,refinement=null;
      if(precisionToggle.checked&&handPrecision.ready()) {
        const refined=await handPrecision.refine(rawHands,els.video,now,performance.now()-started);
        if(epoch!==trackingEpoch||!state.running) {
          if(state.running&&state.stream===stream)state.frameId=requestAnimationFrame(loop);
          return;
        }
        detectedHands=refined.result;refinement=refined.observation;
      }
      const result = handTracker.update(detectedHands,now);
      if (now - state.lastPoseAt >= 90) {
        state.lastPoseAt = now;
        state.pose = state.poseLandmarker.detectForVideo(els.video, now).landmarks[0] || null;
        const tracking = shoulderTracker.update(state.pose, els.video.videoWidth, els.video.videoHeight, now);
        state.shoulders = tracking.reference;
        state.shoulderHeld = tracking.held;
      }
      const observation={now,media_time_ms:mediaTime,hands:rawHands.landmarks,refinement,
        handedness:rawHands.handedness||rawHands.handednesses||[],world_hands:rawHands.worldLandmarks||null,
        pose:state.lastPoseAt===now?state.pose:null,pose_status:state.lastPoseAt===now?(state.pose?"detected":"missing"):"not_run"};
      rawCapture?.observe(observation);sharedMotion?.observe(observation);
      state.slots = assignHandSlots(result, state.slots);
      state.poseArms = poseArmTargets(armTracker.update(state.pose,state.shoulders,state.slots,state.lastPoseAt), state.shoulders, state.slots);
      const armCount = state.poseArms?.filter(Boolean).length || 0;
      els.faceState.textContent = state.shoulderHeld ? "어깨 추적 복귀 대기" : state.shoulders ? "어깨 기준 연결됨" : "어깨 대기";
      els.trackingState.textContent = state.shoulderHeld ? "어깨 복귀 대기 · 출력 보류" : armCount === 2 ? "어깨·양팔 추적" : armCount === 1 ? "한쪽 팔 추적 · 반대쪽 관절 대기" : state.shoulders ? "팔꿈치·손목 추적 대기" : "어깨 대기";
      drawLandmarks(result);
      state.lastVector = state.shoulders && !state.shoulderHeld && !result.tracking.suppressed ? landmarksToVector(result.landmarks, state.shoulders) : null;
      state.fingerspellVector=!result.tracking.suppressed?landmarksToVector(result.landmarks):null;
      if(!state.recording) observeFingerspellFrame(state.fingerspellVector,now);
      const activity = inputActivity.update(motionHands.tracking.suppressed ? [] : motionHands.landmarks, now);
      const jamoActivity = fingerspellActivity.update(result.tracking.suppressed ? [] : result.landmarks, now);
      const rest=restGate.update(state.lastVector,state.samples,now,{moving:activity.moving});
      const wasRestBlocked=state.restActive;
      state.restActive=document.querySelector('#restGuardToggle').checked&&rest.blocked;
      const restMessage=rest.partialMatch?'대기 자세와 유사 · 가려진 손 확인 중':rest.active?'수어 대기 자세':rest.reason==='dtw'?'DTW 대기 자세 확인 중':'대기 자세 확인 중';
      const restReason={samples:'손 개수별 자료 부족',position:'등록한 손 위치와 다름',shape:'정지 자세 기준 미달',tracking:'어깨·손 추적 확인 필요','other-sign':`다른 수어 후보: ${rest.otherLabel||''}`}[rest.reason]||'비교 중';
      const restDetail=`${rest.hands??0}손 추적 · 같은 손 개수 자료 ${rest.compatibleCount??0}개 · 위치 일치 ${rest.nearbyCount??0}개`;
      const restDtw=rest.sequenceLabel?`DTW 후보 ${sampleName(rest.sequenceLabel)} ${Math.round((rest.sequenceSimilarity||0)*100)}점`:'DTW 비교 대기';
      document.querySelector('#restState').textContent=!document.querySelector('#restGuardToggle').checked?'대기 자세 차단 꺼짐':state.restActive?`${restMessage} · 비교 ${Math.round(rest.similarity*100)}점 · 출력 차단`:`대기 자세 ${rest.count}개 · ${restReason} · 정지 비교 ${Math.round(rest.similarity*100)}점 / 기준 75점 · ${restDtw} · ${restDetail}`;
      if(state.restActive&&!wasRestBlocked) {
        tcnStream.pause();state.stabilizer.reset();
        state.motion={lastVector:null,lastAt:0,active:false,quietSince:0,frames:[]};
        if(!yoloOnly()) {
          automaticSpelling.finish();automaticSpelling.pause({lost:true});spelling.resetCandidate();
          fingerspellEpoch++;resetFingerspellFrames();state.fingerspellPreview=null;
        }
        state.sentenceArmed=true;
      }
      if (!yoloOnly() && (activity.moving || jamoActivity.moving) && !state.restActive && automaticEnabled()) automaticSpelling.noteMotion(now);
      // Image classification needs visible hands, not the TCN shoulder reference.
      if (!state.recording && (state.yolo || fingerspelling || automaticEnabled())) {
        if(result.tracking.suppressed && !yoloOnly()) {
          state.fingerspellStill=false;state.fingerspellPreview=null;
          fingerspellEpoch++;spelling.resetCandidate();automaticSpelling.pause();
          els.yoloResult.textContent="지문자 보류: 손 관절 재확인 중입니다.";
        }
        evaluateYolo(result.tracking.suppressed ? [] : result.landmarks, now);
      }
      if (state.lastVector) {
        updatePartialTracking(state.lastVector, now);
        if(state.restActive&&!state.recording) {
          if(tcnEnabled)tcnStream.push(state.lastVector,now,{active:false,moving:false});
          showPrediction(rest.active?'수어 대기':'대기 확인 중',`${restMessage} · 수어·지문자 출력 보류`,0);
        } else if (!fingerspelling && tcnEnabled) {
          if (!state.recording) {
            tcnStream.push(state.lastVector, now, {active:activity.active,moving:activity.moving});
          }
        } else if (!fingerspelling && !state.recording) {
          const dynamicResult = updateMotionRecognition(state.lastVector, now);
          if (dynamicResult) applyRecognition(dynamicResult, "동작");
          else if (!state.motion.active) predict(state.lastVector, now);
        }
      } else {
        tcnStream.missing(now);
        state.partialMatchAllowed = false;
        state.stabilizer.reset();
        const preservingMotion = noteMotionGap(now);
        els.trackingState.textContent = state.shoulderHeld ? "어깨 복귀 대기 · 출력 보류" : state.shoulders ? "손 없음" : "어깨 대기";
        if (preservingMotion) {
          els.trackingState.textContent = "추적 복귀 대기";
          els.confidence.textContent = "짧은 추적 누락입니다. 동작을 잠시 보존하며 결과 확정은 멈춥니다.";
        } else showPrediction("대기 중", state.shoulders ? "손이 감지되지 않았습니다." : "양어깨와 손이 함께 보이도록 해 주세요.", 0);
      }
      captureFrame(now);
      if(!activity.active&&!state.restActive) renderFingerspellPreview(now);
      updateControls();
    }
    if (now - state.lastFrameAt > 500 && (state.lastVector || state.shoulders || state.lastWrists || state.yoloBoxes || state.yoloBusy || state.fingerspellStill)) {
      state.yoloBoxes=null;
      els.canvas.getContext('2d').clearRect(0,0,els.canvas.width,els.canvas.height);
      fingerspellEpoch++;
      spelling.resetCandidate();
      automaticSpelling.pause({lost:true});
      state.fingerspellStill=false;state.fingerspellPreview=null;
      inputActivity.reset();
  fingerspellActivity.reset();
      state.stillness.reset();
      state.lastWrists = null;
      if (fingerspelling || automaticEnabled()) els.yoloResult.textContent = "지문자 대기: 새 카메라 영상이 없습니다.";
      tcnStream.reset();
      state.lastVector = null;
      state.shoulders = null;
      state.shoulderHeld = false;
      shoulderTracker.reset();
      armTracker.reset();
      handTracker.reset();
      resetFingerspellFrames();
      state.fingerspellVector=null;
      state.pose = null;
      state.poseArms = null;
      els.faceState.textContent = "어깨 대기";
      state.slots = [null, null];
      state.stabilizer.reset();
      state.motion = { lastVector: null, lastAt: 0, active: false, quietSince: 0, frames: [] };
      showPrediction("대기 중", "새 카메라 프레임을 기다리고 있습니다.", 0);
      els.trackingState.textContent = "영상 대기";
      updateControls();
    }
    if (state.recording && now >= state.recording.endAt) finishRecording(true);
    state.frameId = requestAnimationFrame(loop);
  } catch {
    stopCamera();
    state.landmarker?.close();
    state.landmarker = null;
    state.poseLandmarker?.close();
    state.poseLandmarker = null;
    notify("손 추적이 중단되었습니다. 카메라를 다시 시작해 주세요.", true);
  }
}

function evaluateYoloOnly(now) {
  if(!state.yolo || state.yoloBusy || now-state.yoloLastAt<150)return;
  state.yoloLastAt=now;state.yoloBusy=true;
  const model=state.yolo,stream=state.stream,epoch=fingerspellEpoch;
  inferYolo11(els.video,model).then(result=>{
    if(!yoloOnly()||state.yolo!==model||state.stream!==stream||epoch!==fingerspellEpoch||!state.running||state.recording)return;
    const completedAt=performance.now(),fresh=completedAt-now<=500;
    state.yoloBoxes=result.detections&&completedAt-now<=1500?{items:result.detections,at:completedAt}:null;
    state.fingerspellStill=fresh;
    if(!fresh){spelling.resetCandidate();automaticSpelling.pause();els.yoloResult.textContent='YOLO 단독 · 이전 영상 결과 보류';return;}
    const strong=isJamo(result.label)&&Number.isFinite(result.confidence)&&result.confidence>=.6;
    const label=strong?result.label:result.label==='기타'?'기타':null;
    let accepted=false;
    if(fingerspelling){accepted=spelling.observe(label,result.confidence,completedAt,{threshold:.6});if(accepted)renderSpelling();}
    else if(automaticEnabled()){
      accepted=automaticSpelling.observe(label,result.confidence,completedAt,{threshold:.6,requireMotion:false});renderSpelling();
    }
    const reason=accepted?'지문자 입력됨':!strong?'YOLO 점수 미달 또는 지문자 아님':(!fingerspelling&&!automaticEnabled())?'자동 입력 꺼짐':'유지 시간 확인 · 같은 글자 중복 방지';
    const message=`YOLO 단독 ${result.label} ${Math.round(result.confidence*100)}% · 관절 비교 안 함 · ${reason}`;
    els.yoloResult.textContent=message;
    state.fingerspellPreview=strong?{label:result.label,confidence:result.confidence,message,at:completedAt}:null;
    renderFingerspellPreview(completedAt);
  }).catch(error=>{
    if(state.yolo!==model||state.stream!==stream||!yoloOnly())return;
    state.yoloBoxes=null;state.fingerspellPreview=null;state.fingerspellStill=false;
    spelling.resetCandidate();automaticSpelling.pause();els.yoloResult.textContent=`YOLO 단독 추론 오류: ${error.message||'모델을 확인해 주세요.'}`;
  }).finally(()=>{state.yoloBusy=false;});
}

function evaluateYolo(hands, now) {
  if(yoloOnly())return evaluateYoloOnly(now);
  if(jointOnly()) state.yoloBoxes=null;
  if (!state.yolo && ((!fingerspelling && !automaticEnabled()) || !jointOnly())) return;
  const wrists = hands.slice(0, 2).map((hand) => hand[0]);
  if (!wrists.length) {
    state.fingerspellStill=false;state.fingerspellPreview=null;
    fingerspellEpoch++;
    spelling.resetCandidate();
    automaticSpelling.pause({lost:true});
    state.stillness.reset();
    state.lastWrists = null;
    els.yoloResult.textContent = "지문자 대기: 손이 감지되지 않았습니다.";
    if (!state.yolo || jointOnly()) return;
  }
  let motion = Infinity;
  if (wrists.length && state.lastWrists?.length === wrists.length) {
    motion = wrists.reduce((total, wrist, index) => total + Math.hypot(wrist.x - state.lastWrists[index].x, wrist.y - state.lastWrists[index].y), 0) / wrists.length;
  }
  state.lastWrists = wrists.map((wrist) => ({ x: wrist.x, y: wrist.y }));
  const observedStill=state.stillness.update(motion, now);
  state.fingerspellStill=observedStill;
  if (!observedStill) {
    fingerspellEpoch++;
    spelling.resetCandidate();
    automaticSpelling.pause();
    els.yoloResult.textContent = "지문자 후보 비교 중 · 입력은 손 안정 후 확인합니다.";
  }
  if (state.yoloBusy || now - state.yoloLastAt < 150) return;
  if(!state.fingerspellVector) {
    fingerspellEpoch++;
    spelling.resetCandidate();automaticSpelling.pause();
    els.yoloResult.textContent="지문자 교차 확인 대기: 손 관절 추적이 필요합니다.";
    if (!state.yolo || jointOnly()) return;
  }
  state.yoloLastAt = now;
  state.yoloBusy = true;
  const model = state.yolo, epoch = fingerspellEpoch, useJoints=jointOnly();
  const jointThreshold=useJoints ? .8 : .3;
  const joints=compareFingerspell(state.fingerspellVector ? [...state.fingerspellVector] : null,fingerspellFrames,state.samples,jointThreshold,{settled:observedStill});
  const observedAt=now, samples=state.samples, stream=state.stream;
  const prediction=useJoints?Promise.resolve({label:joints.status==='match'?(isJamo(joints.label)?joints.label:'기타'):'보류',confidence:joints.similarity}):inferYolo11(els.video, model);
  prediction.then((result) => {
    if (yoloOnly() || state.yolo !== model || stream!==state.stream || useJoints!==jointOnly() || !state.running || state.recording) return;
    // Visualization is independent of joint agreement and sentence commitment.
    // Timestamp completion so ordinary inference latency doesn't erase new boxes.
    const completedAt=performance.now();
    state.yoloBoxes=!useJoints && result.detections && completedAt-observedAt<=1500
      ? {items:result.detections,at:completedAt} : null;
    if(state.restActive || (!fingerspelling && !automaticEnabled())) return;
    if(!state.fingerspellVector || samples!==state.samples || performance.now()-observedAt>500) {
      spelling.resetCandidate();automaticSpelling.pause();
      els.yoloResult.textContent="지문자 보류: 새 관절·영상 비교를 기다립니다.";
      return;
    }
    const inputThreshold=useJoints ? .8 : .6;
    const agreement=useJoints?{accepted:joints.status==='match'&&isJamo(joints.label)&&joints.similarity>=.8,reason:joints.reason||'joint-score'}:fingerspellAgreement(result,joints,{yoloThreshold:inputThreshold,jointThreshold});
    const canCommit=observedStill && state.fingerspellStill && epoch===fingerspellEpoch;
    const label=canCommit?(agreement.accepted?result.label:result.label==="기타"?"기타":null):null;
    const inputStatus=automaticSpelling.status(performance.now());
    let accepted=false;
    if (fingerspelling) {
      accepted=spelling.observe(label, result.confidence, performance.now(),{threshold:inputThreshold});
      if (accepted) renderSpelling();
    } else {
      accepted=automaticSpelling.observe(label, result.confidence, performance.now(),{threshold:inputThreshold});
      renderSpelling();
    }
    const jointLabel=joints.label||joints.nearestLabel||"보류";
    const reasons={samples:"DTW 지문자 샘플 부족",tracking:"관절 추적 대기",conflict:`관절 자세 ${joints.poseLabel||'보류'} / 동작 ${joints.motionLabel||'보류'} 불일치`,weak:"관절 비교 기준 미달",
      'yolo-score':"YOLO 점수 미달",'joint-score':"관절 비교 기준 미달",disagreement:"두 결과 불일치",other:"지문자 아님"};
    const statuses={'motion-required':"일치 · 새 손 움직임 대기",'motion-expired':"일치 · 움직임 확인 시간 만료",'already-entered':"이미 입력한 글자 · 중복 방지",holding:"일치 · 입력 유지 확인"};
    const reason=!canCommit?"입력 보류 · 손 안정 확인 중":!agreement.accepted?reasons[agreement.reason]||"보류":accepted?"지문자 입력됨":fingerspelling?"일치 · 입력 유지 확인":statuses[inputStatus];
    const message=useJoints?`관절 단독 ${jointLabel} ${Math.round(joints.similarity*100)}점 · ${reason}`:`${model.modelName || 'YOLO'} ${result.label} ${Math.round(result.confidence*100)}% · 관절 ${jointLabel} ${Math.round(joints.similarity*100)}점 · ${reason}`;
    els.yoloResult.textContent=message;
    state.fingerspellPreview=isJamo(result.label)&&result.confidence>=.6?{label:result.label,confidence:result.confidence,message,at:performance.now()}:null;
    renderFingerspellPreview(performance.now());
  }).catch((error) => {
    state.yoloBoxes=null;
    els.yoloResult.textContent = `YOLO11 추론 오류: ${error.message || "모델을 확인해 주세요."}`;
  }).finally(() => { state.yoloBusy = false; });
}

function renderFingerspellPreview(now) {
  const preview=state.fingerspellPreview;
  if(!state.running || state.recording || !state.fingerspellStill ||
    (!fingerspelling&&!automaticEnabled()) || !preview || now-preview.at>500) return;
  showPrediction(`지문자 후보 ${preview.label}`,`${preview.message} · 점수는 정확도가 아닙니다`,preview.confidence,preview.label,"교차 비교",true);
}

function resizeCanvas() {
  const rect = els.canvas.getBoundingClientRect();
  const width = Math.round(rect.width * devicePixelRatio);
  const height = Math.round(rect.height * devicePixelRatio);

  if (els.canvas.width !== width || els.canvas.height !== height) {
    els.canvas.width = width;
    els.canvas.height = height;
  }
}

function drawLandmarks(result) {
  const ctx = els.canvas.getContext("2d");
  ctx.clearRect(0, 0, els.canvas.width, els.canvas.height);
  if (!els.overlayToggle.checked) return;
  // Match object-fit: contain, including letterboxing and the shared CSS mirror.
  const scale = Math.min(els.canvas.width / els.video.videoWidth, els.canvas.height / els.video.videoHeight);
  const width = els.video.videoWidth * scale;
  const height = els.video.videoHeight * scale;
  const x = (els.canvas.width - width) / 2;
  const y = (els.canvas.height - height) / 2;
  if (state.yoloBoxes && state.running && !state.recording && !jointOnly() &&
      performance.now()-state.yoloBoxes.at<750) {
    drawYoloBoxes(ctx,state.yoloBoxes.items,{x,y,width,height},devicePixelRatio);
  }
  ctx.lineWidth = 2 * Math.min(devicePixelRatio, 2);
  ctx.strokeStyle = "#65dfb3";
  ctx.fillStyle = "#f5ce91";
  if (state.poseArms) {
    ctx.strokeStyle = "#bda9ff";
    ctx.lineWidth = 3 * Math.min(devicePixelRatio, 2);
    state.poseArms.forEach((arm) => {
      if (!arm) return;
      const points = [arm.shoulder.source, arm.elbow.source, arm.wrist.source];
      ctx.beginPath();
      points.forEach((point, index) => {
        const px = x + point.x * width;
        const py = y + point.y * height;
        if (index) ctx.lineTo(px, py); else ctx.moveTo(px, py);
      });
      ctx.stroke();
    });
    ctx.strokeStyle = "#65dfb3";
  }
  for (const landmarks of result.landmarks) {
    ctx.beginPath();
    for (const connection of HAND_CONNECTIONS) {
      const a = landmarks[connection.start ?? connection[0]];
      const b = landmarks[connection.end ?? connection[1]];
      ctx.moveTo(x + a.x * width, y + a.y * height);
      ctx.lineTo(x + b.x * width, y + b.y * height);
    }
    ctx.stroke();
    for (const p of landmarks) {
      ctx.beginPath();
      ctx.arc(x + p.x * width, y + p.y * height, 3 * Math.min(devicePixelRatio, 2), 0, Math.PI * 2);
      ctx.fill();
    }
  }
}

function addSample(rawLabel) {
  if (fingerspelling) return;
  const label = rawLabel.trim();
  if (!label || label.length > 40 || !state.lastVector || state.recording || !state.running) return;
  commitSamples([...state.samples, { label, vector: [...state.lastVector] }], `${sampleName(label)}: 샘플 1개를 저장했습니다.`);
}

async function recordSequence(options={}) {
  if (fingerspelling) return;
  automaticSpelling.finish();
  fingerspellEpoch++;
  tcnStream.reset();
  if (state.recording) { finishRecording(false); return; }
  const label = options.rest===true?REST_LABEL:els.labelInput.value.trim();
  if (!label || label.length > 40 || !state.running || !state.lastVector) return;
  const shared=Boolean(sharedMotion&&document.querySelector('#shareMotionVideo').checked);
  if(shared) {
    try{await sharedMotion.begin(label,{rest:isRestLabel(label)});}
    catch(error){notify(error.message,true);return;}
    if(!state.running){sharedMotion.cancel();return;}
  }
  state.motion = { lastVector: null, lastAt: 0, active: false, quietSince: 0, frames: [] };
  state.stabilizer.reset();
  state.recording = { label, shared, frames: [], startAt:performance.now(), endAt: performance.now() + 2000, lastAt: -Infinity };
  els.captureProgress.hidden = false;
  els.captureProgress.value = 0;
  updateControls();
}

function captureFrame(now) {
  const recording = state.recording;
  if (!recording) return;
  els.captureProgress.value = Math.min(1, 1 - (recording.endAt - now) / 2000);
  if (now >= recording.startAt && now <= recording.endAt && state.lastVector && now - recording.lastAt >= 80) {
    recording.frames.push([...state.lastVector]);
    recording.lastAt = now;
  }
}

function finishRecording(save) {
  const recording = state.recording;
  if (!recording) return;
  state.recording = null;
  els.captureProgress.hidden = true;
  if (save && recording.frames.length >= 6) {
    if(recording.shared)sharedMotion.finish(recording.frames);
    else commitSamples([...state.samples, { label: recording.label, sequence: recording.frames }], `${sampleName(recording.label)}: 동작 샘플 1개를 저장했습니다.`);
  } else {if(recording.shared)sharedMotion.cancel();notify(save ? "손이 감지되지 않아 저장된 샘플이 없습니다." : "샘플 수집을 취소했습니다.");}
  updateControls();
}

function updatePartialTracking(vector, now) {
  const count = trackedHandCount(vector);
  state.partialMatchAllowed = count >= 1;
}

function updateMotionRecognition(vector, now) {
  const gap = state.motion.missingSince;
  if (gap !== undefined && now - gap > 220) {
    state.motion = { lastVector: null, lastAt: 0, active: false, quietSince: 0, frames: [] };
    state.stabilizer.reset();
  }
  const motion = state.motion;
  const resumed = motion.missingSince !== undefined;
  if (resumed) {
    delete motion.missingSince;
    motion.lastVector = null;
    motion.lastAt = -Infinity;
    motion.quietSince = 0;
    motion.deltas = [];
  }
  if (now - motion.lastAt < 75) return null;
  const delta = motion.lastVector ? motionMagnitude(motion.lastVector, vector) : 0;
  const previous = motion.lastVector;
  const countChanged = previous && trackedHandCount(previous) !== trackedHandCount(vector);
  motion.lastVector = [...vector];
  motion.lastAt = now;
  if (!motion.active) {
    if (delta > 0.035) {
      motion.active = true;
      motion.frames = previous ? [[...previous]] : [];
      motion.previewAt = -Infinity;
      motion.deltas = [];
      state.stabilizer.reset();
      state.sentenceArmed = true;
      showPrediction("동작 비교 중", "저장된 동작 샘플과 비교하고 있습니다.", 0);
    } else return null;
  }
  motion.frames.push([...vector]);
  previewMotion(now);
  // A recovered coordinate is not evidence of motion or stillness across the gap.
  if (resumed || countChanged) {
    motion.quietSince = 0;
    motion.deltas = [];
    return null;
  }
  motion.deltas = [...(motion.deltas || []), delta].slice(-3);
  const meanMotion = motion.deltas.reduce((sum, value) => sum + value, 0) / motion.deltas.length;
  if (motion.deltas.length < 3 || meanMotion > 0.035) {
    motion.quietSince = 0;
    return null;
  }
  if (!motion.quietSince) motion.quietSince = now;
  if (now - motion.quietSince < 180) return null;
  const frames = motion.frames;
  motion.active = false;
  motion.frames = [];
  motion.displayUntil = now + 2000;
  return frames.length >= 6 ? classifySequence(frames, wordSamples()) : null;
}

function noteMotionGap(now) {
  const motion = state.motion;
  if (!motion.active) return false;
  motion.missingSince ??= now;
  motion.quietSince = 0;
  if (now - motion.missingSince <= 220) return true;
  state.motion = { lastVector: null, lastAt: 0, active: false, quietSince: 0, frames: [] };
  return false;
}

function previewMotion(now) {
  const motion = state.motion;
  if (motion.frames.length < 6 || now - (motion.previewAt ?? -Infinity) < 180) return;
  motion.previewAt = now;
  const result = classifySequence(motion.frames.slice(-32), wordSamples());
  const message = result.reason === "incompatible-tracking"
    ? "감지된 손 개수 또는 좌표 형식이 저장 동작과 달라 비교할 수 없습니다. 양손이 보이는지 확인해 주세요."
    : result.reason === "insufficient-motion"
      ? "남은 손의 실제 움직임이 부족합니다. 마지막 자세만으로 동작을 확정하지 않습니다."
      : result.status === "untrained"
      ? "단어별 동작 샘플이 3개 이상 필요합니다."
      : "동작 중 임시 비교 점수입니다. 끝난 동작을 비교한 뒤 확정합니다.";
  showPrediction("동작 비교 중", message, result.similarity, result.label || result.nearestLabel, "동작", true);
}

function predict(vector, now) {
  if(state.restActive)return;
  if (now < (state.motion.displayUntil || 0)) return;
  const staticSamples = staticComparisonSamples(wordSamples());
  const result = classify(vector, staticSamples, { allowPartial: true });
  if (result.status === "untrained" && wordSamples().some(sample => sample.sequence)) {
    const counts = new Map();
    wordSamples().filter(sample => sample.sequence).forEach(sample => counts.set(sample.label, (counts.get(sample.label) || 0) + 1));
    const ready = [...counts.values()].some(count => count >= 3);
    showPrediction(ready ? "동작 대기" : "동작 샘플 부족", ready
      ? "동작이 끝나면 저장된 동작 샘플과 비교합니다."
      : "DTW 동작 비교에는 단어별 동작 샘플이 3개 이상 필요합니다.", 0);
    return;
  }
  applyRecognition(state.stabilizer.update(result, now), "정지");
}

function applyRecognition(result, mode) {
  if(state.restActive||isRestLabel(result.label)||isJamo(result.label)) return;
  if (result.status === "match" && !allowWordCommit(result.label)) {
    state.stabilizer.reset();
    return;
  }
  if (["match", "pending"].includes(result.status) && result.partial && !state.partialMatchAllowed) {
    result = { ...result, status: "unknown", nearestLabel: result.label, reason: "missing-hands" };
  }
  if (result.status === "untrained") {
    state.sentenceArmed = true;
    showPrediction("학습 필요", `${mode} 수어 이름별 샘플을 3개 이상 저장해 주세요.`, 0);
  } else if (result.status === "unknown") {
    state.sentenceArmed = true;
    showPrediction("판단 보류", result.reason === "missing-hands"
      ? "보이는 손이 없어 확정을 보류합니다."
      : result.reason === "insufficient-motion"
      ? "남은 손의 실제 움직임이 부족합니다. 마지막 자세만으로 동작을 확정하지 않습니다."
      : result.reason === "incompatible-tracking"
      ? "감지된 손 개수 또는 좌표 형식이 저장 동작과 달라 비교할 수 없습니다. 양손이 보이는지 확인해 주세요."
      : "저장된 손모양과 다르거나 후보가 비슷합니다.", result.similarity, result.nearestLabel, mode, true);
  } else if (result.status === "pending") {
    state.sentenceArmed = true;
    showPrediction("확인 중", "손모양을 비교하고 있습니다.", result.similarity, result.label, mode, true);
  } else {
    showPrediction(result.label, result.partial ? `${mode} 수어와 유사합니다. 한 손 누락을 보완한 부분 비교입니다.` : `${mode} 수어와 유사합니다.`, result.similarity, result.label, mode);
    if (state.sentenceArmed || state.sentenceWords.at(-1) !== result.label) {
      state.sentenceWords.push(result.label);
      state.sentenceArmed = false;
      rawCapture?.prediction(result.label,"dtw");
      state.polishedSentence = formatSentence(state.sentenceWords, state.speechStyle);
      renderSentence();
    }
  }
}

function clearSentence() {
  resetFingerspellFrames();
  inputActivity.reset();
  fingerspellActivity.reset();
  automaticSpelling.reset();
  automaticWordIndex = null;
  fingerspellEpoch++;
  endSpelling(false);
  protectedWordIndices.clear();
  tcnStream.reset();
  state.sentenceWords = [];
  state.sentenceArmed = true;
  state.polishedSentence = "";
  renderSentence();
}

function setSpeechStyle(style) {
  state.speechStyle = style;
  if (state.sentenceWords.length) polishCurrentSentence();
}

function polishCurrentSentence() {
  state.polishedSentence = formatSentence(state.sentenceWords, state.speechStyle);
  renderSentence();
}

export function formatSentence(words, speechStyle) {
  if (!words.length) return "";
  const opening = speechStyle === "polite" ? { "나": "저는" } : { "나": "나는" };
  const endings = speechStyle === "polite"
    ? { "가다": "가요", "오다": "와요", "먹다": "먹어요", "마시다": "마셔요", "하다": "해요", "보다": "봐요", "있다": "있어요", "없다": "없어요", "알다": "알아요", "모르다": "몰라요", "이다": "예요" }
    : { "가다": "가", "오다": "와", "먹다": "먹어", "마시다": "마셔", "하다": "해", "보다": "봐", "있다": "있어", "없다": "없어", "알다": "알아", "모르다": "몰라", "이다": "야" };
  const sentence = words.map((word, index) => !protectedWordIndices.has(index) && index === 0 ? (opening[word] || word) : word);
  if (!protectedWordIndices.has(sentence.length - 1)) sentence[sentence.length - 1] = endings[sentence.at(-1)] || sentence.at(-1);
  const text = sentence.join(" ").replace(/[.?!]+$/, "");
  return `${text}.`;
}

function renderSentence() {
  els.sentenceText.textContent = state.sentenceWords.length ? state.sentenceWords.join(" ") : "대기 중";
  els.polishedSentenceText.textContent = state.polishedSentence || "대기 중";
  els.clearSentenceBtn.disabled = !state.sentenceWords.length || Boolean(rawCapture?.busy());
  els.polishSentenceBtn.disabled = !state.sentenceWords.length;
}

function showPrediction(label, message, confidence, nearestLabel = null, mode = "", tentative = false) {
  els.nearestSample.textContent = nearestLabel ? `${mode}${tentative ? " 후보" : ""}: ${nearestLabel}` : "비교 샘플 없음";
  els.prediction.textContent = label;
  els.confidence.textContent = message;
  els.confidenceBar.style.width = `${Math.round(confidence * 100)}%`;
  els.similarity.textContent = `${Math.round(confidence * 100)}%`;
}

function updateControls() {
  const label = els.labelInput.value.trim();
  const busy = Boolean(state.recording || rawCapture?.busy() || sharedMotion?.busy() || state.importing || state.saving || state.storageLoading);
  precisionToggle.disabled=busy;
  document.querySelector('#restGuardToggle').disabled=busy;
  document.querySelector('#restPoseSave').disabled=busy||fingerspelling||!state.running||state.lastVector?.length!==130||state.storageBlocked;
  document.querySelector('#restPoseRecord').disabled=busy||fingerspelling||!state.running||state.lastVector?.length!==130||state.storageBlocked;
  document.querySelector('#shareMotionVideo').disabled=busy;
  document.querySelector('#slrtTrainBtn').disabled=busy||!sharedMotion?.canTrain();
  document.querySelector('#slrtSyncBtn').disabled=busy;
  const canCapture = state.running && Boolean(state.lastVector) && label.length > 0 && label.length <= 40 && !state.storageBlocked;
  els.startBtn.disabled = state.starting;
  els.startBtn.textContent = state.starting ? "준비 중" : state.running ? "카메라 중지" : "카메라 시작";
  els.sampleBtn.disabled = fingerspelling || !canCapture || busy;
  els.recordBtn.disabled = fingerspelling || (!state.recording && (!canCapture || busy));
  els.recordBtn.textContent = state.recording ? "수집 취소" : document.querySelector('#shareMotionVideo').checked ? "2초 영상·동작 수집" : "2초 동작 수집";
  els.labelInput.disabled = busy;
  els.clearBtn.disabled = busy || !state.samples.length || state.storageBlocked;
  els.exportBtn.disabled = !state.samples.length;
  els.importBtn.disabled = busy || state.storageBlocked;
  els.undoBtn.disabled = busy || !state.undo;
  els.labels.querySelectorAll("button").forEach((button) => { button.disabled = busy || state.storageBlocked; });
  if(rawCapture?.busy()||sharedMotion?.busy()) {
    modeSelect.disabled=true;trainTcnBtn.disabled=true;els.clearSentenceBtn.disabled=true;
    for(const id of ['autoFingerspell','fingerspellMethod','fingerspellStart','fingerspellFinish','fingerspellCancel','fingerspellBackspace','fingerspellRepeat','yoloLoadBtn','yoloUnloadBtn','yoloModelFile','yoloLabelsFile'])document.querySelector(`#${id}`).disabled=true;
  }
}

function updateSampleUi() {
  renderSpelling();
  els.sampleCount.textContent = String(state.samples.length);

  const counts = new Map();
  const shoulderCounts = new Map();
  const motionCounts = new Map();
  for (const sample of state.samples) {
    counts.set(sample.label, (counts.get(sample.label) || 0) + 1);
    if (sample.vector?.length === 130) shoulderCounts.set(sample.label, (shoulderCounts.get(sample.label) || 0) + 1);
    if (sample.sequence) motionCounts.set(sample.label, (motionCounts.get(sample.label) || 0) + 1);
  }

  els.labels.replaceChildren(
    ...[...counts.entries()].map(([label, count]) => {
      const item = document.createElement("div");
      item.className = "sample-row";
      const name = document.createElement("span");
      name.textContent = sampleName(label);
      const amount = document.createElement("span");
      const shoulderCount = shoulderCounts.get(label) || 0;
      const motionCount = motionCounts.get(label) || 0;
      amount.className = (isRestLabel(label)?count>=3:shoulderCount>=3) ? "sample-ready" : "sample-pending";
      amount.textContent = isRestLabel(label)?`${count}개 · 대기 전용 · 단어 출력 안 함`:`${count}개 · 정지 ${shoulderCount}개 · 동작 ${motionCount}개`;
      const button = document.createElement("button");
      button.className = "icon-button";
      button.textContent = "×";
      if (icons) button.replaceChildren(icons.createElement(icons.X, { width: 16, height: 16, "aria-hidden": "true" }));
      button.title = `${sampleName(label)} 샘플 삭제`;
      button.setAttribute("aria-label", button.title);
      button.addEventListener("click", () => commitSamples(state.samples.filter((sample) => sample.label !== label), `${sampleName(label)} 샘플을 삭제했습니다.`));
      item.append(name, amount, button);
      return item;
    }),
  );
  if(!state.running) {
    const restCount=counts.get(REST_LABEL)||0;
    document.querySelector('#restState').textContent=restCount?`대기 자세 ${restCount}개 · 카메라 시작 후 비교`:'대기 자세 미등록';
  }
  if (!counts.size) {
    const empty = document.createElement("p");
    empty.className = "empty-samples";
    empty.textContent = "저장된 샘플 없음";
    els.labels.append(empty);
  }
  updateControls();
}

function commitSamples(next, message, remember = true) {
  if (state.saving || state.storageLoading) return false;
  if (state.storageBlocked) { notify("기존 저장 데이터를 읽지 못해 변경을 중지했습니다.", true); return false; }
  try {
    const checked = validateSamples(next);
    if (state.fileStorage || state.database) return commitDatabaseSamples(checked, message, remember);
    localStorage.setItem(STORAGE_KEY, JSON.stringify(checked));
    if (remember) state.undo = state.samples;
    state.samples = checked;
    state.stabilizer.reset();
    showPrediction("대기 중", "샘플이 변경되었습니다.", 0);
    updateSampleUi();
    notify(message);
    return true;
  } catch (error) {
    notify(error.name === "QuotaExceededError" ? "저장 공간이 부족합니다. 백업 후 불필요한 샘플을 정리해 주세요." : error.message, true);
    return false;
  }
}

function databaseSamples(database, mode, samples) {
  return new Promise((resolve, reject) => {
    const transaction = database.transaction("samples", mode);
    const store = transaction.objectStore("samples");
    const request = mode === "readonly" ? store.get(STORAGE_KEY) : store.put(samples, STORAGE_KEY);
    transaction.oncomplete = () => resolve(request.result);
    transaction.onabort = () => reject(transaction.error || new Error("샘플 저장을 완료하지 못했습니다."));
    transaction.onerror = () => reject(transaction.error || request.error);
  });
}

async function fileSamples(samples) {
  const response = await fetch("/api/samples", samples === undefined
    ? { cache: "no-store" }
    : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ samples }) });
  if (!response.ok) throw new Error("컴퓨터 파일 저장에 실패했습니다. 로컬 서버 실행과 디스크 공간을 확인해 주세요.");
  return response.json();
}

async function initializeFileStorage() {
  try {
    const saved = await fileSamples();
    if (saved.samples !== null) {
      state.samples = validateSamples(saved.samples);
      state.storageBlocked = false;
      els.notice.hidden = true;
    } else {
      if (typeof indexedDB !== "undefined") await initializeSampleStorage(true);
      if (state.storageBlocked) throw new Error("기존 데이터를 읽지 못했습니다. 백업을 보존해 주세요.");
      state.storageLoading = true;
      await fileSamples(state.samples);
    }
    state.fileStorage = true;
    const location = document.querySelector("#storageLocation");
    if (location) location.textContent = "컴퓨터 파일";
  } catch (error) {
    state.storageBlocked = true;
    notify(`${error.message} run_web.bat으로 실행해 주세요.`, true);
  } finally {
    state.storageLoading = false;
    updateSampleUi();
  }
}

async function initializeSampleStorage(readOnly = false) {
  try {
    const database = await new Promise((resolve, reject) => {
      const request = indexedDB.open("sign-translator", 1);
      request.onupgradeneeded = () => request.result.createObjectStore("samples");
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
      request.onblocked = () => reject(new Error("다른 탭을 닫고 다시 열어 주세요."));
    });
    const saved = await databaseSamples(database, "readonly");
    if (saved !== undefined) {
      state.samples = validateSamples(saved);
      state.storageBlocked = false;
      els.notice.hidden = true;
    } else if (!state.storageBlocked && !readOnly) {
      // Keep the legacy copy untouched until the user explicitly changes data.
      await databaseSamples(database, "readwrite", state.samples);
    }
    state.database = database;
    database.onversionchange = () => {
      database.close();
      if (state.fileStorage) return;
      state.storageBlocked = true;
      updateControls();
      notify("저장소가 변경되었습니다. 페이지를 새로고침해 주세요.", true);
    };
  } catch (error) {
    state.storageBlocked = true;
    notify(`저장소를 열지 못했습니다. 기존 데이터는 유지됩니다. ${error.message}`, true);
  } finally {
    state.storageLoading = false;
    updateSampleUi();
  }
}

async function commitDatabaseSamples(checked, message, remember) {
  state.saving = true;
  updateControls();
  try {
    if (state.fileStorage) await fileSamples(checked);
    else await databaseSamples(state.database, "readwrite", checked);
    if (remember) state.undo = state.samples;
    state.samples = checked;
    state.stabilizer.reset();
    showPrediction("대기 중", "샘플이 변경되었습니다.", 0);
    notify(message);
    return true;
  } catch (error) {
    notify(error.name === "QuotaExceededError" ? "브라우저 저장 공간이 부족합니다. 백업을 저장해 주세요. 기존 샘플은 유지됩니다." : error.message, true);
    return false;
  } finally {
    state.saving = false;
    updateSampleUi();
  }
}

function loadSamples() {
  try {
    return validateSamples(JSON.parse(localStorage.getItem(STORAGE_KEY) || "[]"));
  } catch {
    state.storageBlocked = true;
    notify("기존 저장 데이터를 읽지 못했습니다. 원본 보호를 위해 저장·삭제를 잠갔습니다.", true);
    return [];
  }
}

function clearSamples() {
  els.clearDialog.close();
  if (state.recording || state.importing) return;
  commitSamples([], "모든 샘플을 삭제했습니다. 되돌리기로 복구할 수 있습니다.");
}

async function undoSamples() {
  if (!state.undo || state.recording || state.importing) return;
  if (await commitSamples(state.undo, "직전 변경을 되돌렸습니다.", false)) {
    state.undo = null;
    els.undoBtn.hidden = true;
    updateControls();
  }
}

function notify(message, error = false) {
  els.notice.hidden = false;
  els.notice.dataset.tone = error ? "error" : "info";
  els.noticeText.textContent = message;
  els.undoBtn.hidden = !state.undo;
}

function exportSamples() {
  const blob = new Blob([JSON.stringify({ version: 3, exportedAt: new Date().toISOString(), samples: state.samples }, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `sign-samples-${new Date().toISOString().slice(0, 10)}.json`;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
  notify(`${state.samples.length}개 샘플의 백업 파일을 만들었습니다.`);
}

async function importSamples() {
  const file = els.importFile.files?.[0];
  if (!file || state.recording || state.importing) return;
  state.importing = true;
  updateControls();
  try {
    if (file.size > 64 * 1024 * 1024) throw new Error("64MB 이하의 샘플 파일만 불러올 수 있습니다.");
    const incoming = validateSamples(JSON.parse(await file.text()));
    const occurrences = new Map();
    state.samples.forEach((sample) => {
      const key = JSON.stringify(sample);
      occurrences.set(key, (occurrences.get(key) || 0) + 1);
    });
    const unique = incoming.filter((sample) => {
      const key = JSON.stringify(sample);
      const existing = occurrences.get(key) || 0;
      if (existing) { occurrences.set(key, existing - 1); return false; }
      return true;
    });
    if (state.samples.length + unique.length > MAX_SAMPLES) throw new Error(`샘플은 최대 ${MAX_SAMPLES}개까지 저장할 수 있습니다.`);
    if (!unique.length) notify("추가할 새 샘플이 없습니다.");
    else await commitSamples([...state.samples, ...unique], `${unique.length}개 샘플을 추가했습니다. 기존 샘플도 유지됩니다.`);
  } catch (error) {
    notify(error instanceof SyntaxError ? "JSON 파일을 읽지 못했습니다. 기존 데이터는 유지됩니다." : error.message, true);
  } finally {
    state.importing = false;
    els.importFile.value = "";
    updateControls();
  }
}
