import test from 'node:test';
import assert from 'node:assert/strict';
import {landmarksToVector} from '../frontend/recognition.js';
import {REST_LABEL,compareRestPose,createRestPoseGate} from '../frontend/rest_pose.js';
const hand=x=>Array.from({length:21},(_,i)=>({x:x+i*.002,y:.6-i*.004,z:i*.001}));
const base=[...landmarksToVector([hand(.45),hand(.55)]),.1,1,.12,1];
const poses=()=>Array.from({length:3},()=>({label:REST_LABEL,vector:[...base]}));
test('rest requires independent examples and shoulder-relative position',()=>{
 assert.equal(compareRestPose(base,poses().slice(0,2)).matched,false);
 assert.equal(compareRestPose(base,poses()).matched,true);
 assert.equal(compareRestPose(base.slice(0,126),poses()).matched,false);
 assert.equal(compareRestPose([...base.slice(0,126),.1,1.4,.12,1.4],poses()).reason,'position');
});
test('rest accepts swapped hands but does not fabricate a missing hand',()=>{
 const swapped=[...base.slice(63,126),...base.slice(0,63),...base.slice(128),...base.slice(126,128)];
 assert.equal(compareRestPose(swapped,poses()).matched,true);
 const single=[...base.slice(0,63),...Array(63).fill(0),.1,1,0,0];
 assert.equal(compareRestPose(single,poses()).matched,false);
 assert.equal(compareRestPose(single,poses().map(row=>({...row,vector:single}))).matched,true);
});
test('different fingers at the same body position are not a rest match',()=>{
 const other=[...base];for(const i of [12,24,36,48,60])other[i]+=1;
 assert.equal(compareRestPose(other,poses()).matched,false);
});
test('rest confirmation is short and moving departure releases immediately',()=>{
 const gate=createRestPoseGate(),samples=poses();
 assert.equal(gate.update(base,samples,0).active,false);
 gate.update(base,samples,90);
 assert.equal(gate.update(base,samples,180).entered,true);
 const departure=[...base.slice(0,126),.5,1,.5,1];
 assert.equal(gate.update(departure,samples,225,{moving:true}).left,true);
});
test('quiet wobble has bounded grace and tracking loss releases',()=>{
 const gate=createRestPoseGate(),samples=poses(),other=[...base];other[12]+=5;
 for(const t of [0,90,180])gate.update(base,samples,t);
 assert.equal(gate.update(other,samples,225).active,true);
 assert.equal(gate.update(other,samples,315).active,false);
 for(const t of [405,495,585])gate.update(base,samples,t);
 assert.equal(gate.update(null,samples,630).active,false);
 assert.equal(gate.update(base,[],720).active,false);
});
test('DTW rest evidence uses three recordings, not three frames',()=>{
 const samples=poses().map(row=>({label:row.label,sequence:Array.from({length:20},()=>[...base])}));
 const gate=createRestPoseGate();let result;
 for(let t=0;t<=900;t+=90)result=gate.update(base,samples,t);
 assert.ok(result.sequenceSimilarity>.99);
 assert.equal(compareRestPose(base,samples.slice(0,1)).matched,false);
});
test('an ordinary word named rest is not silently a background class',()=>{
 assert.equal(compareRestPose(base,poses().map(row=>({...row,label:'rest'}))).matched,false);
});
test('ordinary DTW rest match works even when the final static pose fails',()=>{
 const end=[...base];end[12]+=2;
 const samples=poses().map(row=>({label:row.label,sequence:Array.from({length:20},(_,i)=>i===19?end:[...base])}));
 assert.equal(compareRestPose(base,samples).matched,false);
 const gate=createRestPoseGate();let result;
 for(let now=0;now<=1800;now+=90)result=gate.update(base,samples,now);
 assert.equal(result.reason,'dtw');assert.equal(result.active,true);assert.equal(result.blocked,true);
 assert.equal(result.sequenceLabel,REST_LABEL);
 assert.equal(gate.update(null,samples,1890).blocked,false);
});
test('ordinary words compete against rest before DTW can block output',()=>{
 const away=[...base];away[126]+=.8;away[128]+=.8;
 const samples=poses().map(row=>({label:row.label,sequence:Array.from({length:20},(_,i)=>i===19?base:away)}))
   .concat(poses().map(()=>({label:'word',sequence:Array.from({length:20},()=>[...base])})));
 const gate=createRestPoseGate();let result;
 for(let now=0;now<=1800;now+=90)result=gate.update(base,samples,now);
 assert.equal(result.reason,'other-sign');assert.equal(result.blocked,false);
});
test('old DTW rest evidence cannot block a new moving departure',()=>{
 const end=[...base];end[12]+=2;
 const samples=poses().map(row=>({label:row.label,sequence:Array.from({length:20},(_,i)=>i===19?end:[...base])}));
 const gate=createRestPoseGate();
 for(let now=0;now<=1800;now+=90)gate.update(base,samples,now);
 const away=[...base];away[126]+=.8;away[128]+=.8;
 assert.equal(gate.update(away,samples,1840,{moving:true}).blocked,false);
});
test('incompatible single-hand references do not disable the paired-rest veto',()=>{
 const single=[...base.slice(0,63),...Array(63).fill(0),.1,1,0,0];
 const wrong=[...single];wrong[126]=.8;
 const samples=poses().concat(poses().map(row=>({...row,vector:wrong})));
 const result=compareRestPose(single,samples);
 assert.equal(result.matched,false);assert.equal(result.partialMatch,true);
 assert.equal(result.compatibleCount,3);
});
test('a rest candidate vetoes output before the confirmation timer ends',()=>{
 const gate=createRestPoseGate(),samples=poses();
 const pending=gate.update(base,samples,0,{moving:true});
 assert.equal(pending.active,false);assert.equal(pending.blocked,true);
 const departure=[...base.slice(0,126),.5,1,.5,1];
 assert.equal(gate.update(departure,samples,90,{moving:true}).blocked,false);
});
test('one occluded hand can veto output but cannot confirm full rest',()=>{
 const single=[...base.slice(0,63),...Array(63).fill(0),.1,1,0,0];
 const samples=poses(),gate=createRestPoseGate();
 const partial=compareRestPose(single,samples);
 assert.equal(partial.matched,false);assert.equal(partial.partialMatch,true);
 for(const now of [0,90,180,270]) {
   const result=gate.update(single,samples,now);
   assert.equal(result.active,false);assert.equal(result.blocked,true);
 }
 const moved=[...single];moved[126]=.6;
 assert.equal(gate.update(moved,samples,360,{moving:true}).blocked,false);
 assert.equal(compareRestPose(single,samples.slice(0,2)).partialMatch,undefined);
});
