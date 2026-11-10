import test from 'node:test';
import assert from 'node:assert/strict';
import {createArmTracker,poseArmTargets} from '../frontend/recognition.js';
const reference={x:.5,y:.3,scale:.4,aspect:1};
const point=(x,y)=>({x,y,z:0,visibility:.99,presence:.99});
function pose(){
 const p=Array.from({length:33},()=>point(.5,.5));
 p[11]=point(.7,.3);p[12]=point(.3,.3);p[13]=point(.75,.5);p[14]=point(.25,.5);
 p[15]=point(.65,.65);p[16]=point(.35,.65);return p;
}
test('isolated swapped elbows are suppressed without rewriting raw observations',()=>{
 const tracker=createArmTracker(),first=pose();tracker.update(first,reference,[],0);
 const wrong=pose();[wrong[13],wrong[14]]=[wrong[14],wrong[13]];
 const saved=JSON.stringify(wrong);
 for(const now of [90,90,180]) {
   const filtered=tracker.update(wrong,reference,[],now);
   assert.equal(poseArmTargets(filtered,reference),null);
 }
 assert.equal(JSON.stringify(wrong),saved);
 assert.ok(poseArmTargets(tracker.update(first,reference,[],270),reference).every(Boolean));
});
test('continuous genuine arm crossing remains visible',()=>{
 const tracker=createArmTracker();
 for(let i=0;i<=10;i++) {
   const p=pose();p[13].x-=i*.05;p[14].x+=i*.05;p[15].x-=i*.04;p[16].x+=i*.04;
   assert.ok(poseArmTargets(tracker.update(p,reference,[],i*90),reference).every(Boolean));
 }
});
test('rapid coordinated hand and elbow movement is not rejected as an isolated spike',()=>{
 const tracker=createArmTracker();tracker.update(pose(),reference,[],0);
 const p=pose();p[13].x=.3;p[15].x=.2;
 assert.ok(poseArmTargets(tracker.update(p,reference,[],90),reference)[0]);
});
test('a second independent pose frame can confirm an unusual elbow position',()=>{
 const tracker=createArmTracker();tracker.update(pose(),reference,[],0);
 const p=pose();p[13]=point(.5,.8);
 assert.equal(poseArmTargets(tracker.update(p,reference,[],90),reference)[0],null);
 assert.equal(poseArmTargets(tracker.update(p,reference,[],90),reference)[0],null);
 assert.ok(poseArmTargets(tracker.update(p,reference,[],180),reference)[0]);
});
test('shoulder screen ordering cannot switch anatomical arm identities',()=>{
 const p=pose();[p[11].x,p[12].x]=[.3,.7];
 const arms=poseArmTargets(p,reference);
 assert.deepEqual(arms[0].shoulder.source,p[11]);assert.deepEqual(arms[0].elbow.source,p[13]);
});
test('long tracking gaps discard stale elbow history',()=>{
 const tracker=createArmTracker();tracker.update(pose(),reference,[],0);
 const p=pose();[p[13],p[14]]=[p[14],p[13]];
 assert.ok(poseArmTargets(tracker.update(p,reference,[],500),reference).every(Boolean));
 tracker.reset();assert.ok(poseArmTargets(tracker.update(p,reference,[],600),reference).every(Boolean));
});

test('wrong-side hands cannot pull the arm across the torso',()=>{
 const p=pose(),hand=p[16];
 const arms=poseArmTargets(p,reference,[[hand],null]);
 assert.deepEqual(arms[0].wrist.source,p[15]);
 assert.equal(arms[0].wrist.handAttached,false);
});

test('a far ghost hand cannot replace a visible body wrist',()=>{
 const p=pose(),hand=point(.1,.15);
 const arms=poseArmTargets(p,reference,[[hand],null]);
 assert.deepEqual(arms[0].wrist.source,p[15]);
 const nearby=point(.66,.66);
 assert.equal(poseArmTargets(p,reference,[[nearby],null])[0].wrist.handAttached,true);
});

test('missing body wrist and a remote hand leave the arm unconnected',()=>{
 const p=pose();p[15].visibility=0;p[15].presence=0;
 const arms=poseArmTargets(p,reference,[[point(.1,.1)],null]);
 assert.equal(arms[0],null);
});
