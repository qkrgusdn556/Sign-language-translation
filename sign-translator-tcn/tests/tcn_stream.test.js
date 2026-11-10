import test from 'node:test';
import assert from 'node:assert/strict';
import { createWindowDecoder, createTcnStream } from '../frontend/tcn_stream.js';
const scores = label => ({ scores: [{ label, score: 0.95 }, { label: label === 'A' ? 'B' : 'A', score: 0.05 }] });
const movingFrame = i => [...Array.from({length:126}, (_,j) => j < 63 ? j / 63 + i * 0.05 : 0),i*0.05,0,0,0];
test('commits two changing words without a gap and deduplicates overlapping windows', () => {
  const decoder = createWindowDecoder();
  assert.equal(decoder.update(scores('A'), 0).commit, false);
  decoder.update(scores('A'),150);
  assert.equal(decoder.update(scores('A'), 300).commit, true);
  assert.equal(decoder.update(scores('A'), 600).commit, false);
  assert.equal(decoder.update(scores('A'), 600).status, 'match');
  decoder.update(scores('B'), 750);
  decoder.update(scores('B'),900);
  assert.equal(decoder.update(scores('B'), 1050).commit, true);
});

test('stream commits two words in order without a stop or loss between them', async () => {
  const words=[];
  const stream=createTcnStream({infer(sequence){return Promise.resolve(scores(sequence.at(-1)[126]<1.2?'A':'B'));},onResult(result){if(result.commit)words.push(result.label);},onError(error){throw error;}});
  for(let i=0;i<48;i++) {
    stream.push(movingFrame(i),i*75);
    await new Promise(done=>setTimeout(done,0));
  }
  assert.deepEqual(words,['A','B']);
});

test('130-value wrist movement triggers inference even with unchanged fingers', async () => {
  let calls=0;
  const base=movingFrame(0);
  const stream=createTcnStream({infer(){calls++;return Promise.resolve(scores('A'));},onResult(){},onError(error){throw error;}});
  for(let i=0;i<12;i++) stream.push([...base.slice(0,126),i*0.05,0,0,0],i*75);
  await Promise.resolve();
  assert.equal(calls,1);
});
test('weak candidates are withheld and reset enables a repeated word', () => {
  const decoder = createWindowDecoder();
  assert.equal(decoder.update({ scores: [{ label: 'A', score: 0.6 }] }, 0).status, 'unknown');
  decoder.update(scores('A'), 0);
  decoder.update(scores('A'),150);
  assert.equal(decoder.update(scores('A'), 300).commit, true);
  decoder.reset();
  decoder.update(scores('A'), 400);
  decoder.update(scores('A'),550);
  assert.equal(decoder.update(scores('A'), 700).commit, true);
});

test('decoder exposes the tentative label and distinct withholding reasons', () => {
  const decoder = createWindowDecoder();
  const weak = decoder.update({scores:[{label:'A',score:.6},{label:'B',score:.4}]},0);
  assert.equal(weak.label,'A');
  assert.equal(weak.reason,'low-score');
  assert.equal(weak.commit,false);
  const pending=decoder.update(scores('A'),100);
  assert.equal(pending.reason,'holding');
  assert.equal(pending.heldMs,0);
});

test('a word vetoed by fingerspelling can be confirmed later without resetting the stream', () => {
  const decoder=createWindowDecoder();
  decoder.update(scores('A'),0);
  decoder.update(scores('A'),150);
  assert.equal(decoder.update(scores('A'),300).commit,true);
  decoder.reject('A');
  decoder.update(scores('A'),450);
  decoder.update(scores('A'),600);
  assert.equal(decoder.update(scores('A'),750).commit,true);
  assert.equal(decoder.update(scores('A'),900).commit,false);
});

test('one ambiguous observation does not restart a strong candidate or require a pause',()=>{
  const decoder=createWindowDecoder();
  decoder.update(scores('A'),0);
  decoder.update({scores:[{label:'A',score:.6},{label:'B',score:.4}]},150);
  assert.equal(decoder.update(scores('A'),300).commit,false);
  assert.equal(decoder.update(scores('A'),450).commit,true);
  decoder.update(scores('B'),600);
  decoder.update(scores('B'),750);
  assert.equal(decoder.update(scores('B'),900).commit,true);
});

test('conflicting, sparse, or isolated high-score observations do not confirm',()=>{
  const decoder=createWindowDecoder();
  for(let i=0;i<20;i++) assert.equal(decoder.update(scores(i%2?'A':'B'),i*150).commit,false);
  decoder.reset();
  decoder.update(scores('A'),0);
  assert.equal(decoder.update(scores('A'),300).commit,false);
  assert.equal(decoder.update(scores('A'),1500).commit,false);
});

test('inactive pre-roll prevents a second cold-start wait and discards old in-flight output',async()=>{
  let calls=0,resolve,commits=0;
  const stream=createTcnStream({infer(){calls++;return new Promise(done=>{resolve=done;});},onResult(result){if(result.commit)commits++;},onError(error){throw error;}});
  for(let i=0;i<20;i++) stream.push(movingFrame(0),i*75,{active:false});
  await Promise.resolve();
  assert.equal(calls,0);
  for(let i=20;i<25;i++) stream.push(movingFrame(i-19),i*75,{active:true});
  await Promise.resolve();
  assert.equal(calls,1);
  stream.push(movingFrame(5),1875,{active:false});
  resolve(scores('A'));
  await new Promise(done=>setTimeout(done,0));
  assert.equal(commits,0);
});
test('only one inference runs and obsolete responses cannot append a word', async () => {
  let resolve, calls = 0, results = 0;
  const stream = createTcnStream({ infer() { calls++; return new Promise(done => { resolve = done; }); }, onResult() { results++; }, onError(error) { throw error; } });
  for (let i = 0; i < 25; i++) stream.push(movingFrame(i), i * 75);
  await Promise.resolve();
  assert.equal(calls, 1);
  stream.reset();
  resolve(scores('A'));
  await new Promise(done => setTimeout(done, 0));
  assert.equal(results, 0);
});

test('starts predicting at 12 frames rather than waiting for the full 19 frames', async () => {
  let calls = 0, size = 0;
  const stream = createTcnStream({ infer(sequence) { calls++; size = sequence.length; return Promise.resolve(scores('A')); }, onResult() {}, onError(error) { throw error; } });
  for (let i = 0; i < 11; i++) stream.push(movingFrame(i), i * 75);
  await Promise.resolve();
  assert.equal(calls, 0);
  stream.push(movingFrame(11), 11 * 75);
  await Promise.resolve();
  assert.equal(calls, 1);
  assert.equal(size, 12);
});

test('idle and small alternating jitter never invoke the closed-set model', async () => {
  let calls = 0, commits = 0;
  const stream = createTcnStream({infer() { calls++; return Promise.resolve(scores('A')); }, onResult(result) { if(result.commit) commits++; }, onError(error) { throw error; }});
  for (let i = 0; i < 50; i++) stream.push(movingFrame(i % 2 * 0.2), i * 75);
  await new Promise(done => setTimeout(done, 0));
  assert.equal(calls, 0);
  assert.equal(commits, 0);
});

test('a brief hand gap preserves frames but invalidates an in-flight confirmation', async () => {
  let resolve, results = 0, calls = 0;
  const stream = createTcnStream({infer() { calls++; return new Promise(done => {resolve=done;}); }, onResult(result) {if(result.label)results++;}, onError(error) {throw error;}});
  for(let i=0;i<12;i++) stream.push(movingFrame(i), i*75);
  await Promise.resolve();
  stream.missing(850);
  resolve(scores('A'));
  await new Promise(done => setTimeout(done,0));
  assert.equal(results,0);
  stream.push(movingFrame(12),975);
  await Promise.resolve();
  assert.equal(calls,2);
  resolve(scores('A'));
  await new Promise(done => setTimeout(done,0));
});

test('settled hands invalidate a perfect-score response before the next sampling tick',async()=>{
  let resolve,calls=0;
  const results=[];
  const stream=createTcnStream({infer(){calls++;return new Promise(done=>{resolve=done;});},onResult:r=>results.push(r),onError:e=>{throw e;}});
  for(let i=0;i<12;i++)stream.push(movingFrame(i),i*75,{active:true,moving:true});
  await Promise.resolve();
  stream.push(movingFrame(11),826,{active:true,moving:false});
  resolve({scores:[{label:'안녕하세요',score:1}]});
  await new Promise(done=>setTimeout(done,0));
  assert.equal(results.at(-1).reason,'motion-settled');
  assert.equal(results.some(r=>r.label||r.commit),false);
  for(let i=12;i<20;i++)stream.push(movingFrame(11),i*75,{active:true,moving:false});
  assert.equal(calls,1);
  stream.missing(1500);
  assert.equal(results.at(-1).reason,'tracking-missing');
  assert.equal(results.at(-1).similarity,0);
});
