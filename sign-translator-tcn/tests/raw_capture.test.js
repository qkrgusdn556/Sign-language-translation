import test from 'node:test';
import assert from 'node:assert/strict';
import {createRawCapture} from '../frontend/raw_capture.js';

async function environment(run) {
  const old={recorder:globalThis.MediaRecorder,stream:globalThis.MediaStream,reader:globalThis.FileReader};
  const video={readyState:'live',kind:'video'};
  globalThis.MediaRecorder={isTypeSupported:()=>true};
  globalThis.MediaStream=class {constructor(tracks){assert.deepEqual(tracks,[video]);}};
  globalThis.FileReader=class {readAsDataURL(blob){blob.arrayBuffer().then(buffer=>{this.result='data:video/webm;base64,'+Buffer.from(buffer).toString('base64');this.onload();});}};
  let clock=0,recorder;
  const factory=()=>recorder={state:'inactive',start(){this.state='recording';},stop(){this.state='inactive';this.ondataavailable?.({data:new Blob([Buffer.from([0x1a,0x45,0xdf,0xa3]),'frames'])});this.onstop?.();}};
  try {await run({video,stream:{getVideoTracks:()=>[video]},factory,setTime:t=>{clock=t;},now:()=>clock,recorder:()=>recorder});}
  finally {globalThis.MediaRecorder=old.recorder;globalThis.MediaStream=old.stream;globalThis.FileReader=old.reader;}
}
const settle=()=>new Promise(resolve=>setTimeout(resolve,10));

test('recording stores unmirrored source observations and confirmed events only until stop',async()=>environment(async h=>{
  let uploaded,saved;
  const capture=createRawCapture({now:h.now,makeRecorder:h.factory,post:async body=>{uploaded=body;return {clip:{clip_id:'saved'}};},onSaved:c=>{saved=c;}});
  capture.observe({now:0,media_time_ms:1,hands:[]});
  capture.start(h.stream,{consent:true,camera:{mirrored:false}});
  h.setTime(200);capture.observe({now:200,media_time_ms:500,hands:[],pose:null,pose_status:'missing'});
  capture.prediction('ㄱ','jamo','auto-0');h.setTime(400);capture.prediction('가','jamo','auto-0');
  h.setTime(900);capture.stop();capture.prediction('late','tcn');
  await settle();
  assert.equal(capture.phase(),'idle');assert.equal(saved.clip_id,'saved');
  assert.equal(uploaded.metadata.frames.length,1);assert.equal(uploaded.metadata.predictions.length,1);
  assert.equal(uploaded.metadata.predictions[0].label,'가');assert.equal(uploaded.metadata.audio,false);
  assert.equal(uploaded.metadata.duration_ms,900);assert.equal(uploaded.metadata.camera.mirrored,false);
  assert.ok(uploaded.video_base64.length>0);
}));

test('failed upload preserves the identical body for retry and blocks a new capture',async()=>environment(async h=>{
  const calls=[];let failure=true;
  const capture=createRawCapture({now:h.now,makeRecorder:h.factory,post:async body=>{calls.push(body);if(failure)throw Error('network');return {clip:{clip_id:'ok'}};}});
  capture.start(h.stream,{consent:true});h.setTime(1000);capture.stop();await settle();
  assert.equal(capture.phase(),'error');assert.throws(()=>capture.start(h.stream,{}));
  failure=false;await capture.retry();assert.equal(capture.phase(),'idle');
  assert.equal(calls[0],calls[1]);
}));

test('missing camera never starts a recorder',async()=>environment(async h=>{
  const capture=createRawCapture({now:h.now,makeRecorder:h.factory});
  assert.throws(()=>capture.start(null,{}));assert.equal(capture.active(),false);
}));

test('cancelling a partial motion does not upload either a video or a training sample',async()=>environment(async h=>{
  let posts=0;
  const capture=createRawCapture({now:h.now,makeRecorder:h.factory,post:async()=>{posts++;}});
  capture.start(h.stream,{consent:true});h.setTime(500);capture.cancel();await settle();
  assert.equal(posts,0);assert.equal(capture.phase(),'idle');assert.equal(h.recorder().state,'inactive');
}));
