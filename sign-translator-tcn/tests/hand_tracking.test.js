import test from 'node:test';
import assert from 'node:assert/strict';
import {createHandTracker} from '../frontend/hand_tracking.js';
const hand=(x=.5)=>Array.from({length:21},(_,i)=>({x:x+i*.002,y:.6-i*.004,z:i*.001}));
const result=hands=>({landmarks:hands,handedness:hands.map(()=>[{categoryName:'Left',score:.99}])});

test('overlapping copies of one hand are deduplicated without mutating the detector result',()=>{
  const tracker=createHandTracker(),raw=result([hand(),hand(.502)]);
  const before=JSON.stringify(raw);
  const filtered=tracker.update(raw,0);
  assert.equal(filtered.landmarks.length,1);
  assert.equal(filtered.tracking.duplicates,1);
  assert.equal(JSON.stringify(raw),before);
});

test('two nearby but distinct wrists and palms remain two hands',()=>{
  const filtered=createHandTracker().update(result([hand(.5),hand(.55)]),0);
  assert.equal(filtered.landmarks.length,2);
});

test('coincident palm detections with contradictory fingers remain a single hand',()=>{
  const first=hand(),ghost=hand(.501);
  for(const index of [1,2,3,4,6,7,8,10,11,12,14,15,16,18,19,20])ghost[index].x+=.12;
  const raw=result([first,ghost]),saved=JSON.stringify(raw);
  const filtered=createHandTracker().update(raw,0);
  assert.equal(filtered.landmarks.length,1);assert.equal(filtered.tracking.duplicates,1);
  assert.equal(JSON.stringify(raw),saved);
});

test('overlapping fingers do not merge hands with separate palm anchors',()=>{
  const first=hand(),second=hand(.55);
  for(const index of [4,8,12,16,20])second[index]={...first[index]};
  assert.equal(createHandTracker().update(result([first,second]),0).landmarks.length,2);
});

test('static jitter is reduced while real wrist motion follows promptly',()=>{
  const tracker=createHandTracker();tracker.update(result([hand()]),0);
  let error=0;
  for(let i=1;i<40;i++) {
    const filtered=tracker.update(result([hand(.5+(i%2?.003:-.003))]),i*50);
    error+=Math.abs(filtered.landmarks[0][0].x-.5);
  }
  assert.ok(error/39 < .002);
  const moved=tracker.update(result([hand(.65)]),2000).landmarks[0][0].x;
  assert.ok(moved>.63);
});

test('an isolated shape spike is held for display and flagged unsafe for inference; sustained changes resume',()=>{
  const tracker=createHandTracker();tracker.update(result([hand()]),0);
  const spike=hand();spike.slice(5).forEach(p=>{p.x+=.12;});
  const one=tracker.update(result([spike]),50);
  assert.equal(one.tracking.suppressed,true);
  assert.deepEqual(one.landmarks[0],hand());
  const recovered=tracker.update(result([hand()]),100);
  assert.equal(recovered.tracking.suppressed,false);
  assert.equal(tracker.update(result([spike]),150).tracking.suppressed,true);
  assert.equal(tracker.update(result([spike]),200).tracking.suppressed,false);
});

test('missing, invalid, and stale hands are never kept as fresh observations',()=>{
  const tracker=createHandTracker();tracker.update(result([hand()]),0);
  assert.equal(tracker.update(result([]),50).landmarks.length,0);
  const bad=hand();bad[5].x=NaN;
  assert.equal(tracker.update(result([bad]),100).landmarks.length,0);
  const current=tracker.update(result([hand(.7)]),500);
  assert.equal(current.landmarks[0][0].x,.7);
});

test('global hand assignment does not let a reordered first detection steal the nearest track',()=>{
  const tracker=createHandTracker();tracker.update(result([hand(.3),hand(.6)]),0);
  const next=tracker.update(result([hand(.49),hand(.31)]),50);
  assert.deepEqual(next.tracking.handIds,[0,1]);
  assert.ok(next.landmarks[0][0].x<.35);assert.ok(next.landmarks[1][0].x>.45);
});

test('crossing hands retain velocity-based identity even when handedness briefly flips',()=>{
  const tracker=createHandTracker();let next;
  for(let i=0;i<8;i++) {
    const first=hand(.3+i*.05),second=hand(.7-i*.05).map(p=>({...p,y:p.y+.08}));
    const names=i===4?['Right','Left']:['Left','Right'];
    const entries=[first,second].map((points,j)=>({points,category:[{categoryName:names[j],score:.99}]}));
    if(i%2)entries.reverse();
    next=tracker.update({landmarks:entries.map(e=>e.points),handedness:entries.map(e=>e.category)},i*50);
    assert.deepEqual(next.tracking.handIds,[0,1]);
    assert.equal(next.handedness[0][0].categoryName,'Left');
  }
  assert.ok(next.landmarks[0][0].x>.6);assert.ok(next.landmarks[1][0].x<.4);
});

test('briefly occluded hand identity is remembered but no missing coordinates are emitted',()=>{
  const tracker=createHandTracker();tracker.update(result([hand(.3),hand(.7)]),0);
  const single=tracker.update(result([hand(.71)]),50);
  assert.deepEqual(single.tracking.handIds,[1]);assert.equal(single.landmarks.length,1);
  const both=tracker.update(result([hand(.72),hand(.31)]),100);
  assert.deepEqual(both.tracking.handIds,[0,1]);assert.ok(both.landmarks[0][0].x<.35);
});

test('rapid finger movements use less smoothing than small stationary jitter',()=>{
  const tracker=createHandTracker();tracker.update(result([hand()]),0);
  const moved=hand();moved[8].x+=.035;
  const output=tracker.update(result([moved]),50);
  assert.equal(output.tracking.suppressed,false);
  assert.ok(output.landmarks[0][8].x>hand()[8].x+.03);
});
