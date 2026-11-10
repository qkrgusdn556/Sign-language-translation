const MAX_MODEL_BYTES = 128 * 1024 * 1024;

export function parseLabels(text) {
  const cleaned = text.trim();
  if (!cleaned) throw new Error("라벨 파일이 비어 있습니다.");
  let labels;
  try {
    const json = JSON.parse(cleaned);
    labels = Array.isArray(json) ? json : json.names || json.labels;
  } catch {
    labels = cleaned.split(/\r?\n/).map((label) => label.trim()).filter(Boolean);
  }
  if (!Array.isArray(labels) || !labels.length || labels.some((label) => typeof label !== "string" || !label.trim())) {
    throw new Error("라벨은 줄마다 하나씩 쓰거나 JSON 배열이어야 합니다.");
  }
  return labels.map((label) => label.trim());
}

export function softmax(scores) {
  const max = Math.max(...scores);
  const exp = scores.map((score) => Math.exp(score - max));
  const total = exp.reduce((sum, score) => sum + score, 0);
  return exp.map((score) => score / total);
}

export function classifyOutput(values, labels) {
  if (!values || values.length !== labels.length) throw new Error("모델 출력 수와 라벨 개수가 다릅니다.");
  const scores = Array.from(values);
  if (scores.some(value => !Number.isFinite(value))) throw new Error("모델 출력에 유효하지 않은 값이 있습니다.");
  const sum = scores.reduce((total, value) => total + value, 0);
  const probabilities = scores.every(value => value >= 0 && value <= 1) && Math.abs(sum - 1) < 0.001
    ? scores : softmax(scores);
  const index = probabilities.reduce((best, score, i) => score > probabilities[best] ? i : best, 0);
  return { label: labels[index], confidence: probabilities[index], index };
}

export function createStillnessGate(holdMs = 400, maxMotion = 0.09) {
  let stillSince = null;
  return {
    reset() { stillSince = null; },
    update(motion, now) {
      if (!Number.isFinite(motion) || motion > maxMotion) { stillSince = null; return false; }
      if (stillSince === null) stillSince = now;
      return now - stillSince >= holdMs;
    },
  };
}

export function detectionOutput(values, dims, labels, threshold = .5) {
  const channels = 4 + labels.length;
  if (dims.length !== 3 || dims[0] !== 1 || dims[1] !== channels || values.length !== channels * dims[2]) {
    throw new Error('YOLO11 원시 탐지 출력 [1, 4+글자 수, N]이 필요합니다. 모델과 라벨을 확인하세요.');
  }
  const count = dims[2], candidates = [];
  for (let i = 0; i < count; i++) {
    let index = 0, confidence = 0;
    for (let c = 0; c < labels.length; c++) {
      const score = values[(c + 4) * count + i];
      if (!Number.isFinite(score) || score < 0 || score > 1) throw new Error('유효하지 않은 탐지 점수입니다.');
      if (score > confidence) { confidence = score; index = c; }
    }
    if (confidence < threshold) continue;
    const [x,y,w,h] = [0,1,2,3].map(c => values[c * count + i]);
    if (![x,y,w,h].every(Number.isFinite) || w <= 0 || h <= 0) continue;
    candidates.push({label:labels[index], confidence, index, box:[x-w/2,y-h/2,x+w/2,y+h/2]});
  }
  candidates.sort((a,b)=>b.confidence-a.confidence);
  const kept=[];
  for (const candidate of candidates.slice(0,300)) {
    if (kept.some(other=>{
      const a=candidate.box,b=other.box;
      const area=Math.max(0,Math.min(a[2],b[2])-Math.max(a[0],b[0]))*Math.max(0,Math.min(a[3],b[3])-Math.max(a[1],b[1]));
      return area/((a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-area)>.45;
    })) continue;
    kept.push(candidate);
  }
  if (!kept.length || kept.some(item=>item.label!==kept[0].label)) return {label:'보류',confidence:0,index:-1,detections:kept};
  return {...kept[0], detections:kept};
}

export function endToEndDetectionOutput(values, dims, labels, threshold = .5) {
  if (dims.length !== 3 || dims[0] !== 1 || dims[2] !== 6 || values.length !== dims[1]*6) {
    throw new Error('YOLO26 탐지 출력 [1, N, 6]이 필요합니다.');
  }
  const detections=[];
  for(let i=0;i<dims[1];i++) {
    const [x1,y1,x2,y2,confidence,index]=Array.from(values.slice(i*6,i*6+6));
    if (![x1,y1,x2,y2,confidence,index].every(Number.isFinite) || confidence<0 || confidence>1 ||
        !Number.isInteger(index) || index<0 || index>=labels.length) throw new Error('유효하지 않은 YOLO26 탐지 결과입니다.');
    if(confidence<threshold || x2<=x1 || y2<=y1) continue;
    detections.push({label:labels[index],confidence,index,box:[x1,y1,x2,y2]});
  }
  detections.sort((a,b)=>b.confidence-a.confidence);
  if(!detections.length || detections.some(item=>item.label!==detections[0].label))
    return {label:'보류',confidence:0,index:-1,detections};
  return {...detections[0],detections};
}

export function videoDetections(detections, model, videoWidth, videoHeight) {
  if (!videoWidth || !videoHeight) return [];
  const scale = Math.min(model.width/videoWidth, model.height/videoHeight);
  const w = Math.round(videoWidth*scale), h = Math.round(videoHeight*scale);
  const dx = Math.floor((model.width-w)/2), dy = Math.floor((model.height-h)/2);
  return detections.map(item=>({...item, box:item.box.map((n,i)=>Math.max(0,Math.min(1,(n-(i%2?dy:dx))/(i%2?h:w))))}))
    .filter(item=>item.box[2]>item.box[0] && item.box[3]>item.box[1]);
}

export function drawYoloBoxes(ctx, detections, {x,y,width,height}, pixelRatio=1) {
  ctx.save();
  const fontSize = 14*Math.min(pixelRatio,2);
  ctx.font = `600 ${fontSize}px sans-serif`;
  ctx.lineWidth = 2*Math.min(pixelRatio,2);
  for (const item of detections) {
    const [a,b,c,d]=item.box;
    const left=x+a*width, top=y+b*height;
    ctx.strokeStyle='#ffd46a';ctx.strokeRect(left,top,(c-a)*width,(d-b)*height);
    const text=`${item.label} ${Math.round(item.confidence*100)}%`;
    const labelWidth=Math.min(width,ctx.measureText(text).width+12*pixelRatio),labelHeight=fontSize+8*pixelRatio;
    const labelX=Math.max(x,Math.min(left,x+width-labelWidth));
    const labelY=Math.max(y,Math.min(top-labelHeight,y+height-labelHeight));
    ctx.fillStyle='#151a17';ctx.fillRect(labelX,labelY,labelWidth,labelHeight);
    // Counter-mirror lettering because the shared overlay is mirrored by CSS.
    ctx.save();ctx.translate(labelX+labelWidth,labelY);ctx.scale(-1,1);
    ctx.fillStyle='#ffd46a';ctx.textBaseline='top';ctx.fillText(text,6*pixelRatio,4*pixelRatio,Math.max(1,labelWidth-12*pixelRatio));ctx.restore();
  }
  ctx.restore();
}

export async function loadYolo11Classifier(modelFile, labelsFile) {
  if (!modelFile || !labelsFile) throw new Error("YOLO11 ONNX 모델과 라벨 파일을 모두 선택해 주세요.");
  if (modelFile.size > MAX_MODEL_BYTES) throw new Error("128MB 이하 모델만 브라우저에서 불러올 수 있습니다.");
  const text = await labelsFile.text();
  const labels = parseLabels(text);
  let spec = {};
  try { spec = JSON.parse(text); } catch {}
  const task = spec?.task === 'detect' ? 'detect' : 'classify';
  const size = task === 'detect' ? Number(spec.imgsz) : 224;
  if (!Number.isInteger(size) || size < 32 || size > 1024) throw new Error('모델 입력 크기가 올바르지 않습니다.');
  const ort = await import("https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/ort.min.mjs");
  const session = await ort.InferenceSession.create(await modelFile.arrayBuffer(), {
    executionProviders: ["wasm"],
    graphOptimizationLevel: "all",
  });
  const inputName = session.inputNames[0];
  const metadata = Array.isArray(session.inputMetadata) ? session.inputMetadata[0] : session.inputMetadata?.[inputName];
  // Older ONNX Runtime Web versions do not expose input metadata.
  const dims = metadata?.dimensions || metadata?.shape || [1, 3, size, size];
  const height = Number(dims[dims.length - 2]) || size;
  const width = Number(dims[dims.length - 1]) || size;
  if (dims.length !== 4 || (Number(dims[1]) && Number(dims[1]) !== 3) || height > 1024 || width > 1024) throw new Error("[1, 3, H, W] 입력을 가진 YOLO11 분류 ONNX 모델만 지원합니다.");
  return { ort, session, inputName, width, height, labels, task, format:spec.format, modelName:spec.model || 'YOLO11', outputName: session.outputNames[0] };
}

export function imageToTensor(video, model) {
  const canvas = new OffscreenCanvas(model.width, model.height);
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  const scale = Math.min(model.width / video.videoWidth, model.height / video.videoHeight);
  const drawWidth = model.task === 'detect' ? Math.round(video.videoWidth * scale) : video.videoWidth * scale;
  const drawHeight = model.task === 'detect' ? Math.round(video.videoHeight * scale) : video.videoHeight * scale;
  ctx.fillStyle = "rgb(114, 114, 114)";
  ctx.fillRect(0, 0, model.width, model.height);
  ctx.drawImage(video, model.task === 'detect' ? Math.floor((model.width-drawWidth)/2) : (model.width-drawWidth)/2,
    model.task === 'detect' ? Math.floor((model.height-drawHeight)/2) : (model.height-drawHeight)/2, drawWidth, drawHeight);
  const pixels = ctx.getImageData(0, 0, model.width, model.height).data;
  const plane = model.width * model.height;
  const input = new Float32Array(plane * 3);
  for (let i = 0; i < plane; i++) {
    input[i] = pixels[i * 4] / 255;
    input[plane + i] = pixels[i * 4 + 1] / 255;
    input[plane * 2 + i] = pixels[i * 4 + 2] / 255;
  }
  return input;
}

export async function inferYolo11(video, model) {
  const videoWidth=video.videoWidth, videoHeight=video.videoHeight;
  const input = imageToTensor(video, model);
  const outputs = await model.session.run({ [model.inputName]: new model.ort.Tensor("float32", input, [1, 3, model.height, model.width]) });
  const output = outputs[model.outputName];
  if (model.task !== 'detect') return classifyOutput(output.data, model.labels);
  const result=model.format==='yolo26-end2end'
    ? endToEndDetectionOutput(output.data, output.dims, model.labels)
    : detectionOutput(output.data, output.dims, model.labels);
  return {...result,detections:videoDetections(result.detections,model,videoWidth,videoHeight)};
}
