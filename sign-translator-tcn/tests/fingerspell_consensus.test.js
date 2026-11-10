import test from 'node:test';
import assert from 'node:assert/strict';
import {compareFingerspell,fingerspellAgreement} from '../frontend/fingerspell_consensus.js';
const frame=value=>Array.from({length:126},(_,i)=>i<63?value:0);
const rows=(label,vector)=>Array.from({length:3},()=>({label,vector:[...vector]}));

test('relaxed cross-check accepts 65 points but keeps score and label guards',()=>{
  const options={jointThreshold:.65};
  const yolo={label:'ㄱ',confidence:.96};
  const joints={status:'match',label:'ㄱ',similarity:.79};
  assert.equal(fingerspellAgreement(yolo,joints,options).accepted,true);
  assert.equal(fingerspellAgreement(yolo,{...joints,similarity:.65},options).accepted,true);
  assert.equal(fingerspellAgreement(yolo,{...joints,similarity:.649},options).accepted,false);
  assert.equal(fingerspellAgreement(yolo,{...joints,label:'ㄴ'},options).accepted,false);
  assert.equal(fingerspellAgreement({...yolo,confidence:.89},joints,options).accepted,false);
  assert.equal(fingerspellAgreement(yolo,{...joints,status:'unknown'},options).accepted,false);
  const samples=rows('ㄱ',frame(1));
  const probe=frame(1.06);
  const relaxed=compareFingerspell(probe,[],samples,.65);
  assert.equal(relaxed.status,'match');
  assert.ok(relaxed.similarity>=.65 && relaxed.similarity<.8);
  assert.equal(compareFingerspell(probe,[],samples).status,'unknown');
});

test('consensus requires the same jamo and independent score thresholds',()=>{
  const joints={status:'match',label:'ㄱ',similarity:.85};
  assert.equal(fingerspellAgreement({label:'ㄱ',confidence:.95},joints).accepted,true);
  assert.equal(fingerspellAgreement({label:'ㄴ',confidence:1},joints).reason,'disagreement');
  assert.equal(fingerspellAgreement({label:'ㄱ',confidence:.89},joints).accepted,false);
  assert.equal(fingerspellAgreement({label:'ㄱ',confidence:1},{...joints,similarity:.79}).accepted,false);
  assert.equal(fingerspellAgreement({label:'ㄱ',confidence:.85},joints,{yoloThreshold:.8}).accepted,true);
  assert.equal(fingerspellAgreement({label:'ㄱ',confidence:NaN},joints).accepted,false);
  assert.equal(fingerspellAgreement({label:'기타',confidence:1},joints).accepted,false);
});

test('saved static jamo templates match without mutating samples',()=>{
  const samples=[...rows('ㄱ',frame(1)),...rows('ㄴ',frame(2))];
  const before=JSON.stringify(samples);
  const result=compareFingerspell(frame(1),[],samples);
  assert.equal(result.label,'ㄱ');assert.equal(result.source,'pose');
  assert.equal(JSON.stringify(samples),before);
});

test('motion-only recordings supply one terminal pose each and shoulder-free comparison',()=>{
  const samples=Array.from({length:3},()=>({label:'ㄴ',sequence:Array.from({length:8},()=>[...frame(2),.2,.3,0,0])}));
  assert.equal(compareFingerspell(frame(2),[],samples).label,'ㄴ');
  assert.equal(compareFingerspell(frame(2),[],samples.slice(0,2)).reason,'samples');
  assert.equal(compareFingerspell(null,[],samples).reason,'tracking');
});

test('ordinary words remain competing poses and cannot confirm a jamo',()=>{
  const samples=[...rows('ㄱ',frame(1)),...rows('나',frame(2))];
  const joints=compareFingerspell(frame(2),[],samples);
  assert.equal(joints.label,'나');
  assert.equal(fingerspellAgreement({label:'ㄱ',confidence:1},joints).accepted,false);
});

test('confident conflicting motion and terminal pose results are withheld',()=>{
  const samples=[...rows('ㄱ',frame(1)),...Array.from({length:3},()=>({label:'ㄴ',sequence:[...Array.from({length:5},()=>frame(1)),frame(1.5)]}))];
  assert.equal(compareFingerspell(frame(1),Array.from({length:6},()=>frame(1)),samples).reason,'conflict');
});

test('settled H pose is not vetoed by an older conflicting motion window',()=>{
  const samples=[...rows('ㅎ',frame(1)),...Array.from({length:3},()=>({label:'ㄱ',sequence:[...Array.from({length:5},()=>frame(1)),frame(1.5)]}))];
  const sequence=Array.from({length:6},()=>frame(1));
  const moving=compareFingerspell(frame(1),sequence,samples,.3);
  assert.equal(moving.reason,'conflict');
  assert.equal(moving.poseLabel,'ㅎ');assert.equal(moving.motionLabel,'ㄱ');
  const settled=compareFingerspell(frame(1),sequence,samples,.3,{settled:true});
  assert.equal(settled.status,'match');assert.equal(settled.label,'ㅎ');
  assert.equal(fingerspellAgreement({label:'ㅎ',confidence:.65},settled,{yoloThreshold:.6,jointThreshold:.3}).accepted,true);
  assert.equal(fingerspellAgreement({label:'ㄱ',confidence:.99},settled,{yoloThreshold:.6,jointThreshold:.3}).accepted,false);
});

test('wrist stillness does not override conflict while finger shape is changing',()=>{
  const samples=[...rows('ㅎ',frame(1)),...Array.from({length:3},()=>({label:'ㄱ',sequence:[...Array.from({length:5},()=>frame(1)),frame(1.5)]}))];
  const sequence=[frame(1.5),frame(1.3),frame(1.2),frame(1.1),frame(1),frame(1)];
  const result=compareFingerspell(frame(1),sequence,samples,.3,{settled:true});
  assert.equal(result.reason,'conflict');
});

test('missing jamo samples never fall back to YOLO-only confirmation',()=>{
  const joints=compareFingerspell(frame(1),[],rows('word',frame(1)));
  assert.equal(joints.reason,'samples');
  assert.equal(fingerspellAgreement({label:'ㄱ',confidence:1},joints).accepted,false);
});
