import test from "node:test";
import assert from "node:assert/strict";
import vm from "node:vm";
import { readFile } from "node:fs/promises";

const appSource = (await readFile(new URL("../frontend/app.js", import.meta.url), "utf8")) + "\nexport { renderSpelling, evaluateYolo };";
const recognitionSource = await readFile(new URL("../frontend/recognition.js", import.meta.url), "utf8");
const yoloSource = await readFile(new URL("../frontend/yolo.js", import.meta.url), "utf8");
const tcnSource = await readFile(new URL("../frontend/tcn_stream.js", import.meta.url), "utf8");
const spellingSource = await readFile(new URL("../frontend/fingerspell.js", import.meta.url), "utf8");
const activitySource = await readFile(new URL("../frontend/input_activity.js", import.meta.url), "utf8");
const trackingSource = await readFile(new URL("../frontend/hand_tracking.js", import.meta.url), "utf8");
const precisionSource = await readFile(new URL("../frontend/hand_precision.js", import.meta.url), "utf8");
const sharedSource = await readFile(new URL("../frontend/shared_motion.js", import.meta.url), "utf8");
const restSource = await readFile(new URL("../frontend/rest_pose.js", import.meta.url), "utf8");
const consensusSource = await readFile(new URL("../frontend/fingerspell_consensus.js", import.meta.url), "utf8");
const captureSource = await readFile(new URL("../frontend/raw_capture.js", import.meta.url), "utf8");
const hangulSource = await readFile(new URL("../vendor/hangul.js", import.meta.url), "utf8");

test('YOLO boxes survive missing joints, disabled input and 600ms inference without committing text', async()=>{
  let resolve;
  const h=await setup({inferYolo:()=>new Promise(done=>{resolve=done;})});
  h.api.state.running=true;
  h.api.state.yolo={task:'detect',labels:['ㄱ']};
  h.api.state.fingerspellVector=null;
  h.api.state.restActive=true;
  h.elements.get('#autoFingerspell').checked=false;
  h.setClock(1000);
  h.api.evaluateYolo([],1000);
  h.setClock(1600);
  resolve({label:'ㄱ',confidence:.9,detections:[{label:'ㄱ',confidence:.9,box:[.1,.2,.3,.4]}]});
  await Promise.resolve();await Promise.resolve();
  assert.equal(h.api.state.yoloBoxes.items.length,1);
  assert.equal(h.api.state.yoloBoxes.at,1600);
  assert.equal(h.api.automaticSpelling.text(),'');
  assert.equal(h.api.state.sentenceWords.length,0);
});

test('YOLO display discards responses from a previous camera stream', async()=>{
  let resolve;
  const h=await setup({inferYolo:()=>new Promise(done=>{resolve=done;})});
  h.api.state.running=true;h.api.state.yolo={task:'detect',labels:['ㄱ']};
  h.api.state.stream={};h.setClock(1000);
  h.api.evaluateYolo([],1000);
  h.api.state.stream={};h.setClock(1200);
  resolve({label:'ㄱ',confidence:.9,detections:[{label:'ㄱ',box:[.1,.2,.3,.4]}]});
  await Promise.resolve();await Promise.resolve();
  assert.ok(!h.api.state.yoloBoxes);
});
const vector = Array.from({ length: 126 }, (_, i) => i < 63 ? i / 63 : 0);
const sample = () => ({ label: "hello", vector: [...vector] });

test("automatic fingerspelling explains missing model and jamo samples", async () => {
  const h = await setup();
  h.elements.get('#fingerspellMethod').value = 'consensus';
  h.api.renderSpelling();
  assert.match(h.elements.get('#fingerspellStatus').textContent, /YOLO 모델 미연결/);
  assert.match(h.elements.get('#fingerspellStatus').textContent, /지문자 관절 샘플 없음/);
  h.api.state.yolo = {labels:['ㄱ']};
  h.api.state.samples = [{label:'ㄱ',vector:[...vector]}];
  h.api.renderSpelling();
  assert.equal(h.elements.get('#fingerspellStatus').textContent, '자동 지문자 대기');
  h.elements.get('#autoFingerspell').checked = false;
  h.api.renderSpelling();
  assert.equal(h.elements.get('#fingerspellStatus').textContent, '일반 수어 모드');
});

test("one visible hand keeps partial comparison enabled beyond 220ms", async () => {
  const h = await setup();
  const pair=[...vector.slice(0,63),...vector.slice(0,63)];
  h.api.state.samples=Array.from({length:3},()=>({label:"hello",vector:pair}));
  h.api.updatePartialTracking(pair,100);
  h.api.updatePartialTracking(vector,110);
  h.api.predict(vector,110);
  h.api.updatePartialTracking(vector,310);
  h.api.predict(vector,310);
  assert.equal(h.api.state.sentenceWords.length,1);
  h.api.clearSentence();
  h.api.updatePartialTracking(vector,400);
  h.api.predict(vector,400);
  assert.equal(h.api.state.sentenceWords.length,1);
  assert.equal(h.elements.get("#prediction").textContent,"hello");
  h.api.updatePartialTracking(vector,5000);
  assert.equal(h.api.state.partialMatchAllowed,true);
  h.api.updatePartialTracking(null,5100);
  assert.equal(h.api.state.partialMatchAllowed,false);
  h.api.stopCamera();
  assert.equal(h.api.state.partialMatchAllowed,false);
});

test("short tracking gaps retain observed frames without manufacturing coordinates or confirming", async () => {
  const h = await setup();
  const frames = Array.from({length: 8}, (_, i) => [...vector, i * 0.05, 0, 0, 0]);
  frames.forEach((frame, i) => h.api.updateMotionRecognition(frame, 100 + i * 90));
  const count = h.api.state.motion.frames.length;
  assert.equal(h.api.noteMotionGap(760), true);
  assert.equal(h.api.noteMotionGap(850), true);
  assert.equal(h.api.state.motion.frames.length, count);
  assert.equal(h.api.updateMotionRecognition(frames.at(-1), 940), null);
  assert.equal(h.api.state.motion.active, true);
  assert.equal(h.api.state.motion.frames.length, count + 1);
  assert.equal(h.api.state.motion.quietSince, 0);
  assert.equal(h.api.state.motion.deltas.length, 0);
  assert.equal(h.api.state.sentenceWords.length, 0);
  h.api.stopCamera();
  assert.equal(h.api.state.motion.frames.length, 0);
});

test("a long tracking gap cancels the old segment even if detection returns before another missing frame", async () => {
  const h = await setup();
  h.api.updateMotionRecognition([...vector,0,0,0,0], 100);
  h.api.updateMotionRecognition([...vector,0.1,0,0,0], 190);
  assert.equal(h.api.noteMotionGap(250), true);
  assert.equal(h.api.updateMotionRecognition([...vector,0.1,0,0,0], 500), null);
  assert.equal(h.api.state.motion.active, false);
  assert.equal(h.api.state.motion.frames.length, 0);
});

test("tracking loss cannot count as quiet time toward motion completion", async () => {
  const h = await setup();
  const frame = [...vector,0,0,0,0];
  h.api.state.motion = {active:true,lastVector:frame,lastAt:100,quietSince:100,frames:Array.from({length:6},()=>frame),deltas:[0,0,0]};
  h.api.noteMotionGap(190);
  assert.equal(h.api.updateMotionRecognition(frame, 370), null);
  assert.equal(h.api.state.motion.quietSince, 0);
  assert.equal(h.api.updateMotionRecognition(frame, 460), null);
  assert.equal(h.api.state.motion.active, true);
});

test("a single small motion spike does not restart the quiet timer when the recent mean stays low", async () => {
  const h = await setup();
  const frame = [...vector,0,0,0,0];
  h.api.state.motion = {active:true,lastVector:frame,lastAt:100,quietSince:100,frames:Array.from({length:6},()=>frame),deltas:[0.01,0.01,0.01]};
  h.api.state.samples = Array.from({length:3},()=>({label:"hello",sequence:Array.from({length:6},()=>frame)}));
  assert.equal(h.api.updateMotionRecognition([...vector,0.04,0,0,0], 190), null);
  assert.equal(h.api.state.motion.quietSince, 100);
  const result = h.api.updateMotionRecognition([...vector,0.05,0,0,0], 280);
  assert.equal(result.label, "hello");
  assert.equal(h.api.state.motion.active, false);
});

test("motion score updates before motion ends without committing the preview", async () => {
  const h = await setup();
  const frames = Array.from({length: 9}, (_, i) => [...vector, i * 0.05, 0, 0, 0]);
  h.api.state.samples = Array.from({length: 3}, () => ({label: "hello", sequence: frames}));
  frames.forEach((frame, i) => h.api.updateMotionRecognition(frame, 100 + i * 90));
  assert.equal(h.api.state.motion.active, true);
  assert.notEqual(h.elements.get("#similarity").textContent, "0%");
  assert.equal(h.elements.get("#nearestSample").textContent, "동작 후보: hello");
  assert.equal(h.api.state.sentenceWords.length, 0);
});

test("long one-hand motion previews and confirms after movement ends", async () => {
  const h = await setup();
  const frames = Array.from({length: 8}, (_, i) => [...vector, i * 0.05, 0, 0, 0]);
  const twoHands = frames.map(frame => frame.map((v, i) => i >= 63 && i < 126 ? 1 : v));
  h.api.state.samples = Array.from({length: 3}, () => ({label: "hello", sequence: twoHands}));
  frames.forEach((frame, i) => h.api.updateMotionRecognition(frame, 100 + i * 90));
  assert.equal(h.elements.get("#similarity").textContent, "80%");
  assert.equal(h.api.state.sentenceWords.length, 0);
  let result;
  for(let i=8;i<=12;i++) result=h.api.updateMotionRecognition(frames.at(-1),100+i*90)||result;
  assert.equal(result.status,"match");
  assert.equal(result.partial,true);
  h.api.updatePartialTracking(frames.at(-1),1180);
  h.api.applyRecognition(result,"동작");
  assert.equal(h.api.state.sentenceWords.length,1);
});

test("holding a final pose cannot confirm a word that has trained motion samples", async () => {
  const h=await setup();
  const pair=[...vector.slice(0,63),...vector.slice(0,63)];
  h.api.state.samples=[...Array.from({length:3},()=>({label:"hello",vector:pair})),...Array.from({length:3},()=>({label:"hello",sequence:Array.from({length:12},(_,t)=>pair.map((v,i)=>i===30?v+t*.4:v))}))];
  h.api.updatePartialTracking(vector,1000);
  h.api.predict(vector,1000);
  h.api.predict(vector,1500);
  h.api.predict(vector,5000);
  assert.equal(h.api.state.sentenceWords.length,0);
  assert.equal(h.elements.get("#prediction").textContent,"동작 대기");
});

test("nearest sample remains a tentative candidate without appending an unknown word", async () => {
  const h = await setup();
  h.api.applyRecognition({status: "unknown", nearestLabel: "hello", similarity: 0.6}, "동작");
  assert.equal(h.elements.get("#nearestSample").textContent, "동작 후보: hello");
  assert.equal(h.elements.get("#similarity").textContent, "60%");
  assert.equal(h.api.state.sentenceWords.length, 0);
  h.api.applyRecognition({status: "match", label: "hello", similarity: 0.95}, "동작");
  assert.equal(h.elements.get("#nearestSample").textContent, "동작: hello");
  h.api.stopCamera();
  assert.equal(h.elements.get("#nearestSample").textContent, "비교 샘플 없음");
});

test("sampled motion is recognized and its result survives the next static frame", async () => {
  const h = await setup();
  const frames = Array.from({ length: 9 }, (_, i) => [...vector, i * 0.05, 0, 0, 0]);
  h.api.state.samples = Array.from({ length: 3 }, () => ({ label: "hello", sequence: frames }));
  frames.forEach((frame, i) => {
    h.api.updateMotionRecognition(frame, 100 + i * 90);
    h.api.updateMotionRecognition([...vector, i * 0.05 + 0.025, 0, 0, 0], 145 + i * 90);
  });
  let result;
  for (let i = 9; i <= 11; i++) result = h.api.updateMotionRecognition(frames.at(-1), 100 + i * 90) || result;
  assert.equal(result.status, "match");
  assert.equal(result.label, "hello");
  h.api.applyRecognition(result, "motion");
  h.api.predict(frames.at(-1), 1150);
  assert.equal(h.elements.get("#prediction").textContent, "hello");
  assert.equal([...h.api.state.sentenceWords].join(" "), "hello");
});

test("finger-only motion activates DTW with legacy 126-dimensional samples", async () => {
  const h = await setup();
  const frames = Array.from({ length: 8 }, (_, i) => vector.map((v, j) => j === 30 ? v + i * 0.4 : v));
  h.api.state.samples = Array.from({ length: 3 }, () => ({ label: "fingers", sequence: frames }));
  frames.forEach((frame, i) => h.api.updateMotionRecognition(frame, 100 + i * 90));
  assert.equal(h.api.state.motion.active, true);
  let result;
  for (let i = 8; i <= 10; i++) result = h.api.updateMotionRecognition(frames.at(-1), 100 + i * 90) || result;
  assert.equal(result.label, "fingers");
});

test("motion-only data shows motion guidance, not missing static sample guidance", async () => {
  const h = await setup();
  h.api.state.samples = [{ label: "hello", sequence: Array.from({ length: 6 }, (_,t) => vector.map((v,i)=>i===30?v+t*.4:v)) }];
  h.api.predict(vector, 100);
  assert.equal(h.elements.get("#prediction").textContent, "동작 샘플 부족");
  h.api.state.samples.push(...h.api.state.samples, ...h.api.state.samples);
  h.api.predict(vector, 200);
  assert.equal(h.elements.get("#prediction").textContent, "동작 대기");
});

async function setup({ saved = "[]", denied = false, quota = false, faceLandmarks = [], hands = [], poses = [], indexedStore = null, fileStore = null, inferYolo = null } = {}) {
  const elements = new Map();
  const ctx = { clearRect() {}, strokeRect() {}, beginPath() {}, moveTo() {}, lineTo() {}, stroke() {}, arc() {}, fill() {} };
  function element(id = "") {
    return { id, value: id === "#labelInput" ? "hello" : "", textContent: "", hidden: false, disabled: false,
      style: {}, dataset: {}, children: [], checked: true, files: [], currentTime: 1, readyState: 2,
      videoWidth: 960, videoHeight: 540, width: 960, height: 540,
      addEventListener() {}, setAttribute() {}, append(...items) { this.children.push(...items); },
      replaceChildren(...items) { this.children = items; }, querySelectorAll() { return []; },
      getContext() { return ctx; }, getBoundingClientRect() { return { width: 960, height: 540 }; },
      pause() {}, play() { return Promise.resolve(); }, close() {}, showModal() {}, click() {},
    };
  }
  let clock = 0, calls = 0, stopped = 0, detects = 0;
  const tracks = [{ stop() { stopped++; }, addEventListener() {} }];
  const storage = new Map([["sign-translator-tcn-samples", saved]]);
  const indexed = indexedStore && { open() {
    const request = {};
    setTimeout(() => {
      request.result = { close() {}, transaction() {
        const transaction = { objectStore() { return {
          get(key) {
            const result = { result: indexedStore.get(key) };
            setTimeout(() => transaction.oncomplete(), 0);
            return result;
          },
          put(value, key) {
            const result = {};
            setTimeout(() => {
              if (indexedStore.fail) {
                transaction.error = Object.assign(new Error("full"), { name: "QuotaExceededError" });
                transaction.onabort();
              } else {
                indexedStore.set(key, JSON.parse(JSON.stringify(value)));
                transaction.oncomplete();
              }
            }, 0);
            return result;
          },
        }; } };
        return transaction;
      } };
      request.onsuccess();
    }, 0);
    return request;
  } };
  const context = vm.createContext({
    ...(fileStore ? { fetch: async (url, options = {}) => {
      assert.equal(url, "/api/samples");
      if (fileStore.fail) return { ok: false };
      if (options.method === "POST") fileStore.samples = JSON.parse(options.body).samples;
      return { ok: true, json: async () => ({ samples: fileStore.samples ?? null }) };
    } } : {}),
    ...(indexed ? { indexedDB: indexed } : {}),
    console, performance: { now: () => clock }, devicePixelRatio: 1,
    requestAnimationFrame: () => 1, cancelAnimationFrame() {}, setTimeout, clearTimeout, Blob, URL,
    window: { addEventListener() {} },
    document: { hidden: false, addEventListener() {}, createElement: element,
      querySelector(id) { if (!elements.has(id)) elements.set(id, element(id)); return elements.get(id); } },
    navigator: { mediaDevices: { async getUserMedia() {
      calls++;
      if (denied) throw Object.assign(new Error("denied"), { name: "NotAllowedError" });
      return { getTracks: () => tracks, getVideoTracks: () => tracks };
    } } },
    localStorage: { getItem: (key) => storage.get(key), setItem(key, value) {
      if (quota) throw Object.assign(new Error("full"), { name: "QuotaExceededError" });
      storage.set(key, value);
    } },
  });
  const recognition = new vm.SourceTextModule(recognitionSource, { context });
  vm.runInContext(hangulSource, context);
  const spellingModule = new vm.SourceTextModule(spellingSource, { context });
  await spellingModule.link(() => {});
  await spellingModule.evaluate();
  const activityModule = new vm.SourceTextModule(activitySource, {context});
  await activityModule.link(() => {});
  await activityModule.evaluate();
  const trackingModule = new vm.SourceTextModule(trackingSource,{context});
  await trackingModule.link(()=>{});
  await trackingModule.evaluate();
  const precisionModule=new vm.SourceTextModule(precisionSource,{context});
  await precisionModule.link(()=>{});await precisionModule.evaluate();
  await recognition.link(() => {});
  await recognition.evaluate();
  const restModule=new vm.SourceTextModule(restSource,{context});
  await restModule.link(()=>recognition);await restModule.evaluate();
  const consensusModule=new vm.SourceTextModule(consensusSource,{context});
  await consensusModule.link(()=>recognition);
  await consensusModule.evaluate();
  const captureModule=new vm.SourceTextModule(captureSource,{context});
  await captureModule.link(()=>{});await captureModule.evaluate();
  const sharedModule=new vm.SourceTextModule(sharedSource,{context});
  await sharedModule.link(()=>captureModule);await sharedModule.evaluate();
  let yolo = new vm.SourceTextModule(yoloSource, { context });
  await yolo.link(() => {});
  await yolo.evaluate();
  if (inferYolo) {
    const original=yolo;
    yolo=new vm.SyntheticModule(['loadYolo11Classifier','inferYolo11','createStillnessGate','drawYoloBoxes'],function(){
      this.setExport('drawYoloBoxes',original.namespace.drawYoloBoxes);
      this.setExport('loadYolo11Classifier',original.namespace.loadYolo11Classifier);
      this.setExport('createStillnessGate',original.namespace.createStillnessGate);
      this.setExport('inferYolo11',inferYolo);
    },{context});
    await yolo.link(()=>{});
    await yolo.evaluate();
  }
  const tcn = new vm.SourceTextModule(tcnSource, { context });
  await tcn.link(() => recognition);
  await tcn.evaluate();
  const mockVision = new vm.SyntheticModule(["HandLandmarker", "FaceLandmarker", "PoseLandmarker", "FilesetResolver"], function () {
    this.setExport("HandLandmarker", { HAND_CONNECTIONS: [], createFromOptions: async () => ({
      detectForVideo() { detects++; return { landmarks: hands }; }, close() {},
    }) });
    this.setExport("FaceLandmarker", { createFromOptions: async () => ({
      detectForVideo() { return { faceLandmarks }; }, close() {},
    }) });
    this.setExport("PoseLandmarker", { createFromOptions: async () => ({
      detectForVideo() { return { landmarks: poses }; }, close() {},
    }) });
    this.setExport("FilesetResolver", { forVisionTasks: async () => ({}) });
  }, { context });
  await mockVision.link(() => {});
  await mockVision.evaluate();
  const app = new vm.SourceTextModule(appSource + "\nexport { automaticSpelling, handleTcnResult, tcnStream, spelling, startSpelling, endSpelling, state, startCamera, stopCamera, loop, addSample, recordSequence, captureFrame, finishRecording, commitSamples, importSamples, undoSamples, predict, clearSentence, updateMotionRecognition, applyRecognition, noteMotionGap, updatePartialTracking }; export function setTcnTestEnabled(value) { tcnEnabled = value; }", {
    context, importModuleDynamically: async (specifier) => {
      if (specifier.includes("tasks-vision")) return mockVision;
      throw new Error("3D deliberately unavailable in lifecycle tests");
    },
  });
  await app.link((specifier) => specifier.startsWith("./rest_pose.js") ? restModule : specifier.startsWith("./shared_motion.js") ? sharedModule : specifier.startsWith("./hand_precision.js") ? precisionModule : specifier.startsWith("./raw_capture.js") ? captureModule : specifier.startsWith("./fingerspell_consensus.js") ? consensusModule : specifier.startsWith("./hand_tracking.js") ? trackingModule : specifier.startsWith("./input_activity.js") ? activityModule : specifier.startsWith("./fingerspell.js") ? spellingModule : specifier.startsWith("./recognition.js") ? recognition : specifier.startsWith("./tcn_stream.js") ? tcn : yolo);
  await app.evaluate();
  return { api: app.namespace, elements, storage, metrics: () => ({ calls, stopped, detects }),
    setDenied: (value) => { denied = value; }, setFaceLandmarks: (value) => { faceLandmarks = value; }, setPoses: (value) => { poses = value; }, setClock: (value) => { clock = value; } };
}

test("computer file storage migrates browser samples and survives localStorage quota", async () => {
  const saved = JSON.stringify([sample()]);
  const fileStore = {};
  const indexedStore = new Map([["sign-translator-tcn-samples", [sample(), sample()]]]);
  const h = await setup({ saved, fileStore, indexedStore, quota: true });
  assert.equal(h.api.state.fileStorage, true);
  assert.equal(fileStore.samples.length, 2);
  assert.equal(await h.api.commitSamples([sample()], "saved"), true);
  assert.equal(fileStore.samples.length, 1);
  assert.equal(h.storage.get("sign-translator-tcn-samples"), saved);
  assert.equal(indexedStore.get("sign-translator-tcn-samples").length, 2);
});

test("existing computer file wins over browser data and failed writes preserve memory", async () => {
  const fileStore = { samples: [sample(), sample()] };
  const h = await setup({ saved: "{broken", fileStore });
  assert.equal(h.api.state.storageBlocked, false);
  assert.equal(h.api.state.samples.length, 2);
  fileStore.fail = true;
  assert.equal(await h.api.commitSamples([], "clear"), false);
  assert.equal(h.api.state.samples.length, 2);
  assert.equal(fileStore.samples.length, 2);
});

test("IndexedDB migration keeps legacy storage and saves despite localStorage quota", async () => {
  const indexedStore = new Map();
  const saved = JSON.stringify([sample()]);
  const h = await setup({ indexedStore, saved, quota: true });
  assert.equal(h.api.state.samples.length, 1);
  assert.equal(indexedStore.get("sign-translator-tcn-samples").length, 1);
  assert.equal(await h.api.commitSamples([sample(), sample()], "saved"), true);
  assert.equal(indexedStore.get("sign-translator-tcn-samples").length, 2);
  assert.equal(h.storage.get("sign-translator-tcn-samples"), saved);
  await h.api.undoSamples();
  assert.equal(h.api.state.samples.length, 1);
});

test("IndexedDB takes precedence on reload and failed writes retain saved samples", async () => {
  const indexedStore = new Map([["sign-translator-tcn-samples", [sample(), sample()]]]);
  const h = await setup({ indexedStore, saved: "{broken" });
  assert.equal(h.api.state.storageBlocked, false);
  assert.equal(h.api.state.samples.length, 2);
  indexedStore.fail = true;
  assert.equal(await h.api.commitSamples([], "cleared"), false);
  assert.equal(h.api.state.samples.length, 2);
  assert.equal(indexedStore.get("sign-translator-tcn-samples").length, 2);
  assert.equal(h.api.state.saving, false);
});

test("camera denial re-enables start and a retry can start then release the camera", async () => {
  const h = await setup({ denied: true });
  await h.api.startCamera();
  assert.equal(h.elements.get("#startBtn").disabled, false);
  assert.match(h.elements.get("#noticeText").textContent, /권한/);
  h.setDenied(false);
  await h.api.startCamera();
  assert.equal(h.api.state.running, true);
  h.api.stopCamera();
  assert.equal(h.metrics().stopped, 1);
  assert.equal(h.elements.get("#camera").srcObject, null);
});

test("same video frame is detected once and stopping cannot leave a fresh sample", async () => {
  const h = await setup();
  await h.api.startCamera();
  h.api.loop(100);
  h.api.loop(200);
  assert.equal(h.metrics().detects, 1);
  h.api.state.lastVector = vector;
  h.api.stopCamera();
  h.api.addSample("hello");
  assert.equal(h.api.state.samples.length, 0);
});

test("a stalled video invalidates the previous hand pose", async () => {
  const h = await setup();
  await h.api.startCamera();
  h.api.loop(100);
  h.api.state.lastVector = vector;
  h.api.loop(700);
  assert.equal(h.api.state.lastVector, null);
  assert.equal(h.elements.get("#sampleBtn").disabled, true);
});

test("recording saves observed frames as one motion sequence in a single undoable batch", async () => {
  const h = await setup();
  await h.api.startCamera();
  h.api.state.lastVector = vector;
  h.api.recordSequence();
  h.api.captureFrame(0);
  h.api.captureFrame(150);
  h.api.captureFrame(300);
  h.api.captureFrame(450);
  h.api.captureFrame(600);
  h.api.captureFrame(750);
  h.api.state.lastVector = null;
  h.api.captureFrame(900);
  h.api.finishRecording(true);
  assert.equal(h.api.state.samples.length, 1);
  assert.equal(h.api.state.samples[0].sequence.length, 6);
  h.api.undoSamples();
  assert.equal(h.api.state.samples.length, 0);
});

test("stopping during collection discards unfinished data", async () => {
  const h = await setup();
  await h.api.startCamera();
  h.api.state.lastVector = vector;
  h.api.recordSequence();
  h.api.captureFrame(100);
  h.api.stopCamera();
  h.api.finishRecording(true);
  assert.equal(h.api.state.recording, null);
  assert.equal(h.api.state.samples.length, 0);
});

test("backup import preserves repeated samples and reimport does not multiply them", async () => {
  const h = await setup();
  const file = { size: 2000, text: async () => JSON.stringify({ version: 1, samples: [sample(), sample(), sample()] }) };
  h.elements.get("#importFile").files = [file];
  await h.api.importSamples();
  assert.equal(h.api.state.samples.length, 3);
  await h.api.importSamples();
  assert.equal(h.api.state.samples.length, 3);
  h.elements.get("#importFile").files = [{ size: 1, text: async () => "{" }];
  await h.api.importSamples();
  assert.equal(h.api.state.samples.length, 3);
});

test("quota failures and invalid existing data cannot silently overwrite saved samples", async () => {
  const saved = JSON.stringify([sample()]);
  const h = await setup({ saved, quota: true });
  assert.equal(h.api.commitSamples([], "cleared"), false);
  assert.equal(h.api.state.samples.length, 1);
  assert.equal(h.storage.get("sign-translator-tcn-samples"), saved);
  const corrupt = await setup({ saved: "{broken" });
  assert.equal(corrupt.api.state.storageBlocked, true);
  assert.equal(corrupt.api.commitSamples([], "clear"), false);
  assert.equal(corrupt.storage.get("sign-translator-tcn-samples"), "{broken");
});

test("shoulders are required for capture, and shoulder loss invalidates samples", async () => {
  const hand = Array.from({length:21}, (_,i)=>({x:0.7+i*0.002,y:0.5-i*0.005,z:0}));
  const h = await setup({ hands: [hand] });
  await h.api.startCamera();
  h.api.loop(100);
  assert.equal(h.api.state.lastVector, null);
  assert.equal(h.elements.get("#sampleBtn").disabled, true);
  const pose = Array.from({ length: 33 }, () => ({ x: 0.5, y: 0.5, visibility: 0.99, presence: 0.99 }));
  pose[11] = { x: 0.7, y: 0.45, visibility: 0.99, presence: 0.99 };
  pose[12] = { x: 0.3, y: 0.45, visibility: 0.99, presence: 0.99 };
  h.setPoses([pose]);
  h.elements.get("#camera").currentTime++;
  h.api.loop(200);
  assert.equal(h.api.state.lastVector.length, 130);
  assert.equal(h.elements.get("#sampleBtn").disabled, false);
  h.setPoses([]);
  h.elements.get("#camera").currentTime++;
  h.api.loop(300);
  assert.equal(h.api.state.lastVector, null);
  assert.equal(h.elements.get("#sampleBtn").disabled, true);
});

test("one missing elbow keeps the other arm and the hand-based TCN input",async()=>{
  const hand=Array.from({length:21},(_,i)=>({x:.7+i*.002,y:.5-i*.005,z:0}));
  const pose=Array.from({length:33},()=>({x:.5,y:.5,visibility:.99,presence:.99}));
  pose[11].x=.7;pose[12].x=.3;pose[13].visibility=.1;
  const h=await setup({hands:[hand],poses:[pose]});
  await h.api.startCamera();
  h.api.loop(100);
  assert.equal(h.api.state.poseArms[0],null);
  assert.ok(h.api.state.poseArms[1]);
  assert.equal(h.api.state.lastVector.length,130);
  assert.match(h.elements.get('#trackingState').textContent,/한쪽 팔/);
  h.api.stopCamera();
});

test("stable weak shoulders continue, held missing shoulders cannot enter training samples",async()=>{
  const hand=Array.from({length:21},(_,i)=>({x:.7+i*.002,y:.5-i*.005,z:0}));
  const pose=Array.from({length:33},()=>({x:.5,y:.5,visibility:.99,presence:.99}));
  pose[11].x=.7;pose[12].x=.3;
  const h=await setup({hands:[hand],poses:[pose]});
  await h.api.startCamera();
  h.api.loop(100);
  pose[11].visibility=.45;
  h.elements.get('#camera').currentTime++;
  h.api.loop(200);
  assert.equal(h.api.state.lastVector.length,130);
  assert.equal(h.api.state.shoulderHeld,false);
  h.setPoses([]);
  h.elements.get('#camera').currentTime++;
  h.api.loop(300);
  assert.equal(h.api.state.shoulderHeld,true);
  assert.ok(h.api.state.shoulders);
  assert.equal(h.api.state.lastVector,null);
  assert.equal(h.elements.get('#sampleBtn').disabled,true);
  h.elements.get('#camera').currentTime++;
  h.api.loop(500);
  assert.equal(h.api.state.shoulders,null);
  h.api.stopCamera();
});

test("mouth-era stored samples do not lock backup import or overwrite storage on load", async () => {
  const saved = JSON.stringify([{ label: "hello", vector: [...vector, ...Array(10).fill(0.1)] }]);
  const h = await setup({ saved });
  assert.equal(h.api.state.storageBlocked, false);
  assert.equal(h.elements.get("#importBtn").disabled, false);
  assert.equal(h.api.state.samples[0].vector.length, 130);
  assert.equal(h.storage.get("sign-translator-tcn-samples"), saved);
});

test("stable predictions compose a sentence without repeating a held sign", async () => {
  const h = await setup();
  h.api.state.samples = [sample(), sample(), sample()];
  h.api.predict(vector, 0);
  h.api.predict(vector, 199);
  assert.equal(h.api.state.sentenceWords.length, 0);
  h.api.predict(vector, 200);
  h.api.predict(vector, 800);
  assert.equal([...h.api.state.sentenceWords].join(" "), "hello");
  assert.equal(h.elements.get("#sentenceText").textContent, "hello");
  assert.equal(h.elements.get("#polishedSentenceText").textContent, "hello.");
  h.api.clearSentence();
  assert.equal(h.api.state.sentenceWords.length, 0);
  assert.equal(h.elements.get("#clearSentenceBtn").disabled, true);
});

test("sentence polish keeps the original tokens and applies the selected speech style", async () => {
  const h = await setup();
  assert.equal(h.api.formatSentence(["나", "학교", "가다"], "polite"), "저는 학교 가요.");
  assert.equal(h.api.formatSentence(["나", "학교", "가다"], "casual"), "나는 학교 가.");
  assert.equal(h.api.formatSentence(["안녕하세요"], "polite"), "안녕하세요.");
});

test("fingerspelling commits a protected proper noun and cancel never appends", async () => {
  const h = await setup();
  h.api.state.yolo = {labels:["ㄴ", "ㅏ"]};
  h.api.startSpelling();
  for (const [label, at] of [["ㄴ",0],["ㅏ",500]]) {
    h.api.spelling.observe(label,0.99,at);
    h.api.spelling.observe(label,0.99,at+300);
  }
  h.api.endSpelling();
  assert.deepEqual([...h.api.state.sentenceWords],["나"]);
  assert.equal(h.api.formatSentence(h.api.state.sentenceWords,"polite"),"나.");
  h.api.startSpelling();
  h.api.endSpelling(false);
  assert.equal(h.api.state.sentenceWords.length,1);
  h.api.clearSentence();
  assert.equal(h.api.formatSentence(["나"],"polite"),"저는.");
});

test("automatic fingerspelling keeps words and letters ordered without manual entry", async () => {
  const h=await setup();
  h.api.state.yolo={labels:['ㄱ','ㄴ','기타']};
  h.api.state.running=true;
  h.api.setTcnTestEnabled(true);
  h.api.handleTcnResult({status:'match',label:'안녕하세요',commit:true,similarity:.99});
  h.api.automaticSpelling.noteMotion(100);
  for(const time of [100,250,400,550]) h.api.automaticSpelling.observe('ㄱ',.99,time);
  h.setClock(550);
  h.api.handleTcnResult({status:'match',label:'잘못된겹침',commit:true,similarity:.99});
  assert.deepEqual([...h.api.state.sentenceWords],['안녕하세요','ㄱ']);
  h.api.automaticSpelling.noteMotion(700);
  for(const time of [700,850,1000,1150]) h.api.automaticSpelling.observe('ㄴ',.99,time);
  h.setClock(1800);
  h.api.handleTcnResult({status:'match',label:'감사합니다',commit:true,similarity:.99});
  assert.deepEqual([...h.api.state.sentenceWords],['안녕하세요','ㄱㄴ','감사합니다']);
  h.api.clearSentence();
  assert.equal(h.api.automaticSpelling.text(),'');
  assert.equal(h.api.state.sentenceWords.length,0);
});

test("camera loop runs TCN and automatic YOLO together and discards stopped-camera responses", async () => {
  let yoloCalls=0, resolve;
  const hand=Array.from({length:21},(_,i)=>({x:.7+i*.002,y:.5-i*.005,z:0}));
  const pose=Array.from({length:33},()=>({x:.5,y:.5,visibility:.99,presence:.99}));
  pose[11]={x:.7,y:.45,visibility:.99,presence:.99};
  pose[12]={x:.3,y:.45,visibility:.99,presence:.99};
  const h=await setup({hands:[hand],poses:[pose],inferYolo(){yoloCalls++;return new Promise(done=>{resolve=done;});}});
  await h.api.startCamera();
  h.api.state.yolo={labels:['ㄱ','기타']};
  h.api.setTcnTestEnabled(true);
  let pushes=0;
  h.api.tcnStream.push=(vector,now,{active})=>{if(active)pushes++;};
  for(let time=100;time<=1400;time+=100) {
    if(time<=700) hand.forEach(p=>{p.x+=.025;});
    h.setClock(time);
    h.elements.get('#camera').currentTime++;
    h.api.loop(time);
  }
  assert.ok(pushes > 0);
  assert.ok(pushes < 14);
  assert.equal(yoloCalls,1);
  h.api.stopCamera();
  resolve({label:'ㄱ',confidence:.99});
  await Promise.resolve();
  await Promise.resolve();
  assert.equal(h.api.automaticSpelling.text(),'');
  assert.equal(h.api.state.sentenceWords.length,0);
});

test("automatic fingerspelling continues without shoulders while TCN stays paused", async () => {
  let calls = 0, pushes = 0;
  const hand = Array.from({length:21},(_,i)=>({x:.7+i*.002,y:.5-i*.005,z:0}));
  const h = await setup({hands:[hand], inferYolo:async()=>{calls++;return {label:'ㄱ',confidence:.99};}});
  const shape=hand.flatMap(p=>[(p.x-hand[0].x)/Math.hypot(hand[9].x-hand[0].x,hand[9].y-hand[0].y),(p.y-hand[0].y)/Math.hypot(hand[9].x-hand[0].x,hand[9].y-hand[0].y),0]);
  while(shape.length<126)shape.push(0);
  h.api.state.samples=Array.from({length:3},()=>({label:'ㄱ',vector:[...shape]}));
  await h.api.startCamera();
  h.api.state.yolo = {labels:['ㄱ','기타']};
  h.api.setTcnTestEnabled(true);
  h.api.tcnStream.push = ()=>{pushes++;};
  for (let time=100; time<=2100; time+=100) {
    if(time<=700) hand.forEach(p=>{p.x+=.025;});
    h.setClock(time);
    h.elements.get('#camera').currentTime++;
    h.api.loop(time);
    for (let i=0;i<5;i++) await Promise.resolve();
  }
  assert.ok(calls >= 4);
  assert.equal(h.api.state.lastVector,null);
  assert.equal(pushes,0);
  assert.equal(h.api.automaticSpelling.text(),'ㄱ');
  assert.match(h.elements.get('#yoloResult').textContent,/ㄱ/);
  h.setClock(2800);
  h.api.loop(2800);
  assert.match(h.elements.get('#yoloResult').textContent,/새 카메라 영상/);
  h.api.stopCamera();
});

test("idle tracking noise never requests TCN or types a high-confidence jamo", async () => {
  let pushes=0,calls=0;
  const hand=Array.from({length:21},(_,i)=>({x:.5+i*.003,y:.5-i*.003,z:0}));
  const pose=Array.from({length:33},()=>({x:.5,y:.5,visibility:.99,presence:.99}));
  pose[11].x=.7;pose[12].x=.3;
  const h=await setup({hands:[hand],poses:[pose],inferYolo:async()=>{calls++;return {label: calls%6<3?'ㄴ':'ㄱ',confidence:1};}});
  await h.api.startCamera();
  h.api.state.yolo={labels:['ㄱ','ㄴ','기타']};
  h.api.setTcnTestEnabled(true);
  h.api.tcnStream.push=(vector,now,{active})=>{if(active)pushes++;};
  for(let i=0;i<100;i++) {
    hand.forEach((p,j)=>{p.x=.5+j*.003+(i%2?.004:-.004);p.z=Math.sin(i+j)*.08;});
    h.setClock(i*100+100);
    h.elements.get('#camera').currentTime++;
    h.api.loop(i*100+100);
    for(let n=0;n<5;n++) await Promise.resolve();
  }
  assert.ok(calls>10);
  assert.equal(pushes,0);
  assert.equal(h.api.state.sentenceWords.length,0);
  h.api.stopCamera();
});

test("manual spelling rejects mismatched joint and YOLO labels, then accepts agreement",async()=>{
  const hand=Array.from({length:21},(_,i)=>({x:.7+i*.002,y:.5-i*.005,z:0}));
  let label='ㄱ';
  const h=await setup({hands:[hand],inferYolo:async()=>({label,confidence:.99})});
  const size=Math.hypot(hand[9].x-hand[0].x,hand[9].y-hand[0].y);
  const shape=hand.flatMap(p=>[(p.x-hand[0].x)/size,(p.y-hand[0].y)/size,0]);
  while(shape.length<126)shape.push(0);
  h.api.state.samples=Array.from({length:3},()=>({label:'ㄴ',vector:[...shape]}));
  await h.api.startCamera();
  h.api.state.yolo={labels:['ㄱ','ㄴ','기타']};
  h.api.startSpelling();
  for(let time=100;time<=1400;time+=100) {
    h.setClock(time);h.elements.get('#camera').currentTime++;h.api.loop(time);
    for(let i=0;i<5;i++)await Promise.resolve();
  }
  assert.equal(h.api.spelling.text(),'');
  assert.match(h.elements.get('#yoloResult').textContent,/두 결과 불일치/);
  label='ㄴ';
  for(let time=1500;time<=2300;time+=100) {
    h.setClock(time);h.elements.get('#camera').currentTime++;h.api.loop(time);
    for(let i=0;i<5;i++)await Promise.resolve();
  }
  assert.equal(h.api.spelling.text(),'ㄴ');
  h.api.stopCamera();
});

test("general DTW and TCN cannot bypass jamo consensus",async()=>{
  const h=await setup();
  h.api.state.running=true;
  h.api.setTcnTestEnabled(true);
  h.api.handleTcnResult({status:'match',label:'ㄱ',similarity:1,commit:true});
  h.api.applyRecognition({status:'match',label:'ㄴ',similarity:1},'동작');
  assert.equal(h.api.state.sentenceWords.length,0);
});

test("a static G appears as a candidate without silently bypassing the motion guard",async()=>{
  const hand=Array.from({length:21},(_,i)=>({x:.7+i*.002,y:.5-i*.005,z:0}));
  const h=await setup({hands:[hand],inferYolo:async()=>({label:'ㄱ',confidence:.99})});
  const size=Math.hypot(hand[9].x-hand[0].x,hand[9].y-hand[0].y);
  const shape=hand.flatMap(p=>[(p.x-hand[0].x)/size,(p.y-hand[0].y)/size,0]);
  while(shape.length<126)shape.push(0);
  h.api.state.samples=Array.from({length:3},()=>({label:'ㄱ',vector:[...shape]}));
  await h.api.startCamera();
  h.api.state.yolo={labels:['ㄱ','기타']};
  for(let time=100;time<=1500;time+=100) {
    h.setClock(time);h.elements.get('#camera').currentTime++;h.api.loop(time);
    for(let i=0;i<5;i++)await Promise.resolve();
  }
  assert.match(h.elements.get('#prediction').textContent,/지문자 후보 ㄱ/);
  assert.match(h.elements.get('#yoloResult').textContent,/새 손 움직임 대기/);
  assert.equal(h.api.state.sentenceWords.length,0);
  for(let time=1600;time<=3500;time+=100) {
    if(time<=2200)hand.forEach(p=>{p.x+=.025;});
    h.setClock(time);h.elements.get('#camera').currentTime++;h.api.loop(time);
    for(let i=0;i<5;i++)await Promise.resolve();
  }
  assert.equal(h.api.automaticSpelling.text(),'ㄱ');
  h.api.stopCamera();
});

test('small A movement enters one letter after agreement without arming TCN',async()=>{
  const hand=Array.from({length:21},(_,i)=>({x:.5+i*.002,y:.5-i*.005,z:0}));
  const h=await setup({hands:[hand],inferYolo:async()=>({label:'ㅏ',confidence:.62})});
  const size=Math.hypot(hand[9].x-hand[0].x,hand[9].y-hand[0].y);
  const shape=hand.flatMap(p=>[(p.x-hand[0].x)/size,(p.y-hand[0].y)/size,0]);
  while(shape.length<126)shape.push(0);
  h.api.state.samples=Array.from({length:3},()=>({label:'ㅏ',vector:[...shape]}));
  await h.api.startCamera();h.api.state.yolo={labels:['ㅏ','기타']};
  h.api.setTcnTestEnabled(true);
  let pushes=0;h.api.tcnStream.push=(_vector,_now,options)=>{if(options.active)pushes++;};
  for(let time=100;time<=3000;time+=100) {
    if(time>=200&&time<=500)hand.forEach(p=>{p.x+=.006;});
    h.setClock(time);h.elements.get('#camera').currentTime++;h.api.loop(time);
    for(let i=0;i<5;i++)await Promise.resolve();
  }
  assert.equal(h.api.state.sentenceWords.join(''),'ㅏ');
  assert.equal(pushes,0);
  h.api.stopCamera();
});

test('stable H reaches the sentence despite a conflicting motion template',async()=>{
  const hand=Array.from({length:21},(_,i)=>({x:.5+i*.002,y:.5-i*.005,z:0}));
  const h=await setup({hands:[hand],inferYolo:async()=>({label:'ㅎ',confidence:.65})});
  const size=Math.hypot(hand[9].x-hand[0].x,hand[9].y-hand[0].y);
  const shape=hand.flatMap(p=>[(p.x-hand[0].x)/size,(p.y-hand[0].y)/size,0]);
  while(shape.length<126)shape.push(0);
  const other=shape.map((v,i)=>i<63?v+.5:v);
  h.api.state.samples=[...Array.from({length:3},()=>({label:'ㅎ',vector:[...shape]})),
    ...Array.from({length:3},()=>({label:'ㄱ',sequence:[...Array.from({length:5},()=>[...shape]),other]}))];
  await h.api.startCamera();h.api.state.yolo={labels:['ㅎ','ㄱ','기타']};
  for(let time=100;time<=3000;time+=100) {
    if(time>=200&&time<=500)hand.forEach(p=>{p.x+=.006;});
    h.setClock(time);h.elements.get('#camera').currentTime++;h.api.loop(time);
    for(let i=0;i<5;i++)await Promise.resolve();
  }
  assert.equal(h.api.state.sentenceWords.join(''),'ㅎ');
  h.api.stopCamera();
});

test('YOLO-only automatic input works with no hands or samples and does not repeat a held letter',async()=>{
  const h=await setup({hands:[],inferYolo:async()=>({label:'ㅎ',confidence:.65})});
  h.elements.get('#fingerspellMethod').value='yolo';
  h.api.state.samples=[];
  await h.api.startCamera();h.api.state.yolo={labels:['ㅎ','기타']};
  for(let time=100;time<=3000;time+=100){
    h.setClock(time);h.elements.get('#camera').currentTime++;h.api.loop(time);
    for(let i=0;i<5;i++)await Promise.resolve();
  }
  assert.equal(h.api.state.sentenceWords.join(''),'ㅎ');
  assert.match(h.elements.get('#yoloResult').textContent,/YOLO 단독.*관절 비교 안 함/);
  assert.doesNotMatch(h.elements.get('#fingerspellStatus').textContent,/관절 샘플 없음/);
  h.api.stopCamera();
});

test('YOLO-only manual input needs no joints and commits on finish',async()=>{
  const h=await setup({hands:[],inferYolo:async()=>({label:'ㅏ',confidence:.7})});
  h.elements.get('#fingerspellMethod').value='yolo';h.api.state.samples=[];
  await h.api.startCamera();h.api.state.yolo={labels:['ㅏ','기타']};h.api.startSpelling();
  for(let time=100;time<=1200;time+=100){
    h.setClock(time);h.elements.get('#camera').currentTime++;h.api.loop(time);
    for(let i=0;i<5;i++)await Promise.resolve();
  }
  assert.equal(h.api.spelling.text(),'ㅏ');
  assert.equal(h.api.state.sentenceWords.length,0);
  h.api.endSpelling();assert.equal(h.api.state.sentenceWords.join(''),'ㅏ');
  h.api.stopCamera();
});

test('YOLO-only still rejects low scores, stopped cameras and mode-switched responses',async()=>{
  const h=await setup({hands:[],inferYolo:async()=>({label:'ㅎ',confidence:.59})});
  h.elements.get('#fingerspellMethod').value='yolo';await h.api.startCamera();h.api.state.yolo={labels:['ㅎ']};
  for(let time=100;time<=1500;time+=100){
    h.setClock(time);h.elements.get('#camera').currentTime++;h.api.loop(time);
    for(let i=0;i<5;i++)await Promise.resolve();
  }
  assert.equal(h.api.state.sentenceWords.length,0);h.api.stopCamera();
  for(const change of ['stop','mode']){
    let resolve;
    const pending=await setup({hands:[],inferYolo:()=>new Promise(done=>{resolve=done;})});
    pending.elements.get('#fingerspellMethod').value='yolo';await pending.api.startCamera();pending.api.state.yolo={labels:['ㅎ']};
    pending.setClock(200);pending.elements.get('#camera').currentTime++;pending.api.loop(200);
    if(change==='stop')pending.api.stopCamera();else pending.elements.get('#fingerspellMethod').value='consensus';
    resolve({label:'ㅎ',confidence:.99});for(let i=0;i<5;i++)await Promise.resolve();
    assert.equal(pending.api.state.sentenceWords.length,0);
    pending.api.stopCamera();
  }
});

test('quiet sequence samples can confirm a held DTW word',async()=>{
  const h=await setup();
  const pose=[...vector,.2,.1,0,0];
  h.api.state.samples=Array.from({length:3},()=>({label:'held',sequence:Array.from({length:20},()=>[...pose])}));
  h.api.predict(pose,100);h.api.predict(pose,400);
  assert.equal(h.elements.get('#prediction').textContent,'held');
  assert.equal(h.api.state.sentenceWords.length,1);
});

test('rest vetoes perfect TCN and DTW results and cannot become an output word',async()=>{
  const h=await setup();
  h.api.state.running=true;
  h.api.setTcnTestEnabled(true);
  h.api.state.restActive=true;
  h.api.handleTcnResult({status:'match',label:'hello',similarity:1,commit:true});
  h.api.applyRecognition({status:'match',label:'hello',similarity:1},'motion');
  assert.equal(h.api.state.sentenceWords.length,0);
  h.api.state.restActive=false;
  h.api.handleTcnResult({status:'match',label:'__rest__',similarity:1,commit:true});
  h.api.applyRecognition({status:'match',label:'__rest__',similarity:1},'motion');
  assert.equal(h.api.state.sentenceWords.length,0);
});

test('joint-only fingerspelling does not require or call YOLO',async()=>{
  const hand=Array.from({length:21},(_,i)=>({x:.7+i*.002,y:.5-i*.005,z:0}));let calls=0;
  const h=await setup({hands:[hand],inferYolo:async()=>{calls++;throw Error('YOLO should not run');}});
  const size=Math.hypot(hand[9].x-hand[0].x,hand[9].y-hand[0].y);
  const shape=hand.flatMap(p=>[(p.x-hand[0].x)/size,(p.y-hand[0].y)/size,0]);while(shape.length<126)shape.push(0);
  h.api.state.samples=Array.from({length:3},()=>({label:'ㅂ',vector:shape}));
  h.elements.get('#fingerspellMethod').value='joints';
  await h.api.startCamera();h.api.startSpelling();
  for(let time=100;time<=1500;time+=100) {
    h.setClock(time);h.elements.get('#camera').currentTime++;h.api.loop(time);
    for(let i=0;i<5;i++)await Promise.resolve();
  }
  assert.equal(calls,0);assert.equal(h.api.spelling.text(),'ㅂ');
  assert.match(h.elements.get('#yoloResult').textContent,/관절 단독/);
  h.api.stopCamera();
});
