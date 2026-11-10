import test from 'node:test';
import assert from 'node:assert/strict';
import {handCrop,mapCropHand,acceptRefinement,createHandPrecision} from '../frontend/hand_precision.js';
const hand=(x=.45)=>Array.from({length:21},(_,i)=>({x:x+i*.002,y:.6-i*.004,z:i*.001}));
const video={videoWidth:960,videoHeight:540};
const raw=()=>({landmarks:[hand()],handedness:[[{categoryName:'Left',score:.99}]]});
const local=(points,crop)=>points.map(p=>({x:(p.x*crop.width-crop.x)/crop.size,y:(p.y*crop.height-crop.y)/crop.size,z:p.z*crop.width/crop.size}));

test('square pixel crop maps XY and depth back to the unmirrored source',()=>{
  const original=hand(),crop=handCrop(original,960,540);
  assert.ok(crop);
  const mapped=mapCropHand(local(original,crop),crop);
  for(let i=0;i<21;i++)for(const key of ['x','y','z'])assert.ok(Math.abs(mapped[i][key]-original[i][key])<1e-10);
  assert.equal(handCrop(hand(-.1),960,540),null);
  assert.equal(handCrop(hand(),0,0),null);
  assert.equal(mapCropHand([{x:0,y:0,z:0}],crop),null);
});

test('a different wrist or implausible shape is not substituted',()=>{
  assert.equal(acceptRefinement(hand(),hand()),true);
  assert.equal(acceptRefinement(hand(),hand(.8)),false);
  const bad=hand();bad[9].y-=.2;
  assert.equal(acceptRefinement(hand(),bad),false);
});

function fixture({delay=0,timeoutMs=35,detected,bitmap}={}) {
  let worker,posts=0,closed=0;
  const service=createHandPrecision({timeoutMs,supported:()=>true,
    makeWorker:()=>worker={terminate(){closed++;},postMessage({id}){posts++;setTimeout(()=>worker.onmessage({data:{id,result:detected||{landmarks:[local(hand(),handCrop(hand(),960,540))]}}}),delay);}},
    bitmap:bitmap||(async()=>({close(){}}))});
  service.start();worker.onmessage({data:{type:'ready'}});
  return {service,worker,posts:()=>posts,closed:()=>closed};
}

test('accepted same-frame refinement preserves raw observations and rate limits extra inference',async()=>{
  const {service,posts}=fixture(),input=raw(),before=JSON.stringify(input);
  const output=await service.refine(input,video,0,5);
  assert.equal(output.observation.accepted,true);
  assert.notEqual(output.result,input);
  assert.equal(JSON.stringify(input),before);
  assert.equal((await service.refine(input,video,100,5)).observation,null);
  assert.equal(posts(),1);service.stop();
});

test('overlapping hands and slow full-frame inference skip enlargement',async()=>{
  const {service,posts}=fixture();
  assert.equal((await service.refine({landmarks:[hand(),hand(.46)]},video,0,5)).observation,null);
  assert.equal((await service.refine(raw(),video,1000,25)).observation,null);
  assert.equal(posts(),0);service.stop();
});

test('late inference falls back without using the response or queuing more work',async()=>{
  const {service,posts}=fixture({delay:60,timeoutMs:5}),input=raw();
  const output=await service.refine(input,video,0,1);
  assert.equal(output.result,input);assert.equal(output.observation.accepted,false);
  assert.equal(output.observation.reason,'deadline');assert.equal(service.ready(),false);
  await service.refine(input,video,2000,1);
  assert.equal(posts(),1);
  await new Promise(r=>setTimeout(r,70));
  assert.equal(service.ready(),true);service.stop();
});

test('stopping closes late bitmaps and cannot send to a restarted worker',async()=>{
  let finish,closed=0;
  const {service,posts}=fixture({bitmap:()=>new Promise(r=>finish=r)});
  const response=service.refine(raw(),video,0,1);await Promise.resolve();
  service.stop();finish({close(){closed++;}});
  assert.equal((await response).observation.accepted,false);
  await Promise.resolve();assert.equal(posts(),0);assert.equal(closed,1);
});

test('conflicting handedness or multiple detected hands reject the crop result',async()=>{
  for(const detected of [{landmarks:[local(hand(),handCrop(hand(),960,540))],handedness:[[{categoryName:'Right',score:.99}]]},{landmarks:[hand(),hand()]}]) {
    const {service}=fixture({detected});const input=raw(),output=await service.refine(input,video,0,1);
    assert.equal(output.result,input);assert.equal(output.observation.accepted,false);service.stop();
  }
});
