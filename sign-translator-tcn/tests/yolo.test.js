import test from "node:test";
import assert from "node:assert/strict";
import { parseLabels, softmax, classifyOutput, createStillnessGate, detectionOutput, endToEndDetectionOutput, videoDetections, drawYoloBoxes } from "../frontend/yolo.js";

test('YOLO26 end-to-end boxes preserve xyxy and reject conflicting hands', () => {
  const labels=['ㄱ','ㄴ'];
  const result=endToEndDetectionOutput([10,20,80,100,.9,0,0,0,0,0,0,0],[1,2,6],labels);
  assert.equal(result.label,'ㄱ');
  assert.deepEqual(result.box,[10,20,80,100]);
  assert.equal(result.detections.length,1);
  assert.equal(endToEndDetectionOutput([10,20,80,100,.9,0,100,20,180,100,.8,1],[1,2,6],labels).label,'보류');
  assert.equal(endToEndDetectionOutput([10,20,80,100,.4,0],[1,1,6],labels).label,'보류');
  assert.throws(()=>endToEndDetectionOutput([10,20,80,100,.9,2],[1,1,6],labels));
  assert.throws(()=>endToEndDetectionOutput([10,20,80,100,NaN,0],[1,1,6],labels));
  assert.throws(()=>endToEndDetectionOutput([10,20,80,100,.9,0],[1,6,1],labels));
});

test("YOLO labels accept plain text and common JSON shapes", () => {
  assert.deepEqual(parseLabels("hello\nthank you\n"), ["hello", "thank you"]);
  assert.deepEqual(parseLabels('{"names":["a","b"]}'), ["a", "b"]);
  assert.throws(() => parseLabels(""));
  assert.throws(() => parseLabels('{"names":[]}'));
});

test("YOLO classifier output produces a normalized winning class", () => {
  const probabilities = softmax([1, 2, 3]);
  assert.ok(Math.abs(probabilities.reduce((sum, p) => sum + p, 0) - 1) < 1e-12);
  const result = classifyOutput(new Float32Array([0, 3, 1]), ["a", "b", "c"]);
  assert.equal(result.label, "b");
  assert.ok(result.confidence > 0.8);
  assert.throws(() => classifyOutput(new Float32Array([1]), ["a", "b"]));
});

test("YOLO runs only after hands remain still and resets when motion returns", () => {
  const gate = createStillnessGate(400, 0.09);
  assert.equal(gate.update(Infinity, 0), false);
  assert.equal(gate.update(0.05, 100), false);
  assert.equal(gate.update(0.05, 499), false);
  assert.equal(gate.update(0.05, 500), true);
  assert.equal(gate.update(0.12, 520), false);
  assert.equal(gate.update(0.05, 921), false);
});

test("exported probability outputs are not softened a second time", () => {
  assert.ok(classifyOutput([0.01,0.99],["ㄱ","ㄴ"]).confidence > .98);
  assert.throws(() => classifyOutput([NaN,1],["ㄱ","ㄴ"]));
});

test('detection raw output rejects missing targets and conflicting hands',()=>{
  const labels=['ㄱ','ㄴ'];
  assert.equal(detectionOutput([50,50,20,20,.9,.1],[1,6,1],labels).label,'ㄱ');
  assert.equal(detectionOutput([50,50,20,20,.1,.1],[1,6,1],labels).label,'보류');
  assert.equal(detectionOutput([50,150,50,150,20,20,20,20,.9,.1,.1,.9],[1,6,2],labels).label,'보류');
  assert.equal(detectionOutput([50,51,50,51,20,20,20,20,.95,.1,.1,.9],[1,6,2],labels).label,'ㄱ');
  assert.throws(()=>detectionOutput([50,50,20,20,NaN,.1],[1,6,1],labels));
  assert.throws(()=>detectionOutput([.9,.1],[1,2],labels));
  assert.deepEqual(parseLabels('{"task":"detect","imgsz":320,"names":["ㄱ","ㄴ"]}'),labels);
});

test('boxes remove model padding, clip to video and discard padding-only boxes',()=>{
  const rows=videoDetections([{box:[80,115,240,205]},{box:[0,0,320,320]},{box:[0,0,20,30]}],{width:320,height:320},1280,720);
  assert.deepEqual(rows.map(r=>r.box),[[.25,.25,.75,.75],[0,0,1,1]]);
  assert.deepEqual(videoDetections([{box:[115,80,205,240]}],{width:320,height:320},720,1280)[0].box,[.25,.25,.75,.75]);
});

test('overlay uses contained video coordinates and counter-mirrors text',()=>{
  const calls=[];
  const ctx=new Proxy({measureText:()=>({width:60})},{get:(target,key)=>key in target?target[key]:(...args)=>calls.push([key,...args])});
  drawYoloBoxes(ctx,[{box:[.25,.25,.75,.75],label:'ㄱ',confidence:.91}],{x:10,y:20,width:800,height:450});
  assert.deepEqual(calls.find(c=>c[0]==='strokeRect'),['strokeRect',210,132.5,400,225]);
  assert.deepEqual(calls.find(c=>c[0]==='scale'),['scale',-1,1]);
  assert.equal(calls.find(c=>c[0]==='fillText')[1],'ㄱ 91%');
});
