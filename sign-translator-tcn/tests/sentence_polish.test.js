import test from 'node:test';
import assert from 'node:assert/strict';
import vm from 'node:vm';
import {readFileSync} from 'node:fs';
import {assembleJamoText,createSentencePolisher,requestConfirmedSentence} from '../frontend/sentence_polish.js';
const context={window:{}};
vm.runInNewContext(readFileSync(new URL('../vendor/hangul.js',import.meta.url),'utf8'),context);
const pause=()=>new Promise(resolve=>setTimeout(resolve,10));

test('upstream temporary errors retry once, while timeout, quota and local busy do not',async()=>{
  let calls=0;const delays=[];
  const response=(ok,error)=>({ok,json:async()=>ok?{sentence:'보정 완료'}:{error}});
  const result=await requestConfirmedSentence({},undefined,{
    fetchImpl:async()=>++calls===1?response(false,'서버 오류 (HTTP 503, gemini-3.6-flash)'):response(true),
    wait:async ms=>delays.push(ms)});
  assert.equal(result.sentence,'보정 완료');assert.equal(calls,2);assert.deepEqual(delays,[3200]);
  for(const message of ['다른 문장을 처리 중입니다.','응답 대기 시간 초과(20초)','사용량 초과 (HTTP 429)','인증 실패 (HTTP 403)']) {
    calls=0;
    await assert.rejects(requestConfirmedSentence({},undefined,{fetchImpl:async()=>{calls++;return response(false,message);},wait:async()=>{}}));
    assert.equal(calls,1);
  }
  calls=0;
  await assert.rejects(requestConfirmedSentence({},undefined,{fetchImpl:async()=>{calls++;return response(false,'HTTP 503');},wait:async()=>{}}));
  assert.equal(calls,2);
});

test('jamo assembles deterministically without guessing names',()=>{
  assert.equal(assembleJamoText('안녕하세요 만나서 반가워요 나/저 이름 ㅂ ㅏ ㄱ ㅎ ㅕ ㄴ ㅇ ㅜ',letters=>context.window.Hangul.assemble(letters)),
    '안녕하세요 만나서 반가워요 나/저 이름 박현우');
});

test('returned sentences join separated jamo without inventing missing syllables',()=>{
  const assemble=letters=>context.window.Hangul.assemble(letters);
  assert.equal(assembleJamoText('내 이름은 ㅂ ㅏ ㄱ ㅇ ㅜ야.',assemble),'내 이름은 박우야.');
  assert.equal(assembleJamoText('제 이름은 ㅂ ㅏ ㄱ ㅎ ㅕ ㄴ ㅇ ㅜ입니다.',assemble),'제 이름은 박현우입니다.');
  assert.equal(assembleJamoText('제 이름은 박현우입니다.',assemble),'제 이름은 박현우입니다.');
});

test('automatic correction waits for idle and sends latest input without a button',async()=>{
  let calls=0;const results=[];
  const p=createSentencePolisher({request:async s=>{calls++;assert.equal(s.raw,'추가 입력');return {sentence:'확정 결과'};},onResult:t=>results.push(t),onStatus:()=>{},delayMs:40,intervalMs:0});
  p.update({raw:'입력',enabled:false});p.update({raw:'입력',enabled:true});
  p.update({raw:'추가 입력',enabled:true});await pause();assert.equal(calls,0);
  await new Promise(resolve=>setTimeout(resolve,50));
  assert.equal(calls,1);assert.deepEqual(results,['확정 결과']);p.dispose();
});

test('new input discards stale output and automatically queues the latest text',async()=>{
  const calls=[],results=[];
  const p=createSentencePolisher({request:(snapshot,signal)=>new Promise(resolve=>calls.push({snapshot,signal,resolve})),onResult:t=>results.push(t),onStatus:()=>{},delayMs:0,intervalMs:0});
  p.update({raw:'확정 전',enabled:true},{force:true});await pause();
  p.update({raw:'새 입력',enabled:true});calls[0].resolve({sentence:'오래된 결과'});await pause();
  assert.equal(calls[0].signal.aborted,false);assert.equal(calls.length,2);assert.deepEqual(results,[]);
  assert.equal(calls[1].snapshot.raw,'새 입력');calls[1].resolve({sentence:'새 확정 결과'});await pause();
  assert.deepEqual(results,['새 확정 결과']);p.dispose();
});

test('confirmed requests serialize; repeated clicks do not duplicate active request',async()=>{
  const calls=[],results=[];
  const p=createSentencePolisher({request:(snapshot,signal)=>new Promise(resolve=>calls.push({snapshot,signal,resolve})),onResult:t=>results.push(t),onStatus:()=>{},intervalMs:0});
  p.update({raw:'첫 확정',enabled:true},{force:true});await pause();
  p.update({raw:'첫 확정',enabled:true},{force:true});await pause();assert.equal(calls.length,1);
  p.update({raw:'두 번째 확정',enabled:true},{force:true});await pause();assert.equal(calls.length,1);
  // MutationObserver can report the same text after the confirmation click.
  p.update({raw:'두 번째 확정',enabled:true});
  calls[0].resolve({sentence:'이전'});await pause();assert.equal(calls.length,2);
  assert.equal(calls[1].snapshot.raw,'두 번째 확정');assert.deepEqual(results,[]);
  calls[1].resolve({sentence:'최신'});await pause();assert.deepEqual(results,['최신']);p.dispose();
});

test('clear or disable prevents late results and drops queued confirmations',async()=>{
  for(const disable of [true,false]) {
    const calls=[],results=[];
    const p=createSentencePolisher({request:s=>new Promise(resolve=>calls.push({s,resolve})),onResult:t=>results.push(t),onStatus:()=>{},intervalMs:0});
    p.update({raw:'첫 확정',enabled:true},{force:true});await pause();
    p.update({raw:'다음 확정',enabled:true},{force:true});
    if(disable)p.update({raw:'다음 확정',enabled:false});else p.dispose();
    calls[0].resolve({sentence:'취소된 결과'});await pause();assert.deepEqual(results,[]);assert.equal(calls.length,1);p.dispose();
  }
});

test('failure preserves local text and explicit confirmation can retry',async()=>{
  let calls=0;const results=[],statuses=[];
  const p=createSentencePolisher({request:async()=>{if(++calls===1)throw Error('연결 실패');return {sentence:'성공'};},onResult:t=>results.push(t),onStatus:t=>statuses.push(t),intervalMs:0});
  p.update({raw:'원문',enabled:true},{force:true});await pause();assert.deepEqual(results,[]);assert.equal(statuses.at(-1),'연결 실패');
  p.update({raw:'원문',enabled:true},{force:true});await pause();assert.deepEqual(results,['성공']);p.dispose();
});
