import test from 'node:test';
import assert from 'node:assert/strict';
import {staticComparisonSamples,classify,landmarksToVector} from '../frontend/recognition.js';
const hand=Array.from({length:21},(_,i)=>({x:.4+i*.002,y:.6-i*.004,z:0}));
const vector=[...landmarksToVector([hand]),.2,.1,0,0];
const recordings=()=>Array.from({length:3},()=>({label:'held',sequence:Array.from({length:20},()=>[...vector])}));
test('three quiet recordings support static recognition without source mutation',()=>{
 const samples=recordings(),before=JSON.stringify(samples);
 const result=classify(vector,staticComparisonSamples(samples));
 assert.equal(result.label,'held');assert.equal(result.status,'match');
 assert.equal(JSON.stringify(samples),before);
});
test('slow travel and return-to-start gestures cannot become static words',()=>{
 const samples=recordings();
 samples[0].sequence.forEach((frame,i)=>{frame[126]+=.02*Math.min(i,19-i);});
 assert.deepEqual(staticComparisonSamples(samples),[]);
});
test('tracking count changes and incomplete recordings cannot authorize static output',()=>{
 const samples=recordings();samples[0].sequence[5][70]=1;
 assert.deepEqual(staticComparisonSamples(samples),[]);
 assert.equal(classify(vector,staticComparisonSamples(recordings().slice(0,2))).status,'untrained');
});
test('a missing hand cannot confirm a held two-hand recording',()=>{
 const pair=[...vector.slice(0,63),...vector.slice(0,63),.2,.1,.3,.1];
 const samples=recordings().map(row=>({...row,sequence:row.sequence.map(()=>pair)}));
 assert.notEqual(classify(vector,staticComparisonSamples(samples),{allowPartial:true}).status,'match');
});
