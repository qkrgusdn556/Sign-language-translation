import test from 'node:test';
import assert from 'node:assert/strict';
import {createSentenceSpeaker,isSpeechShortcut} from '../frontend/sentence_speech.js';

function setup(options={}) {
  let snapshot={raw:'안녕하세요',text:'안녕하세요.'};
  const utterances=[],statuses=[];let clears=0;
  class Utterance{constructor(text){this.text=text;}}
  const speaker=createSentenceSpeaker({synthesis:{getVoices:()=>[{lang:'ko-KR',localService:true,name:'한국어'}],speak:u=>utterances.push(u),cancel(){}},Utterance,
    getSnapshot:()=>snapshot,onStarted:()=>{clears++;snapshot={raw:'대기 중',text:'대기 중'};},onStatus:m=>statuses.push(m),...options});
  return {speaker,utterances,statuses,setSnapshot:s=>{snapshot=s;},clears:()=>clears};
}

test('speaks the displayed final sentence and clears only when Korean speech starts',()=>{
  const h=setup();
  assert.equal(h.speaker.speak(),true);
  assert.equal(h.utterances[0].text,'안녕하세요.');assert.equal(h.utterances[0].lang,'ko-KR');
  assert.equal(h.utterances[0].voice.localService,true);assert.equal(h.clears(),0);
  assert.equal(h.speaker.speak(),false);assert.equal(h.utterances.length,1);
  h.utterances[0].onstart();assert.equal(h.clears(),1);
  h.utterances[0].onend();assert.equal(h.speaker.busy(),false);
});

test('unavailable speech and errors before starting preserve the sentence',()=>{
  const missing=setup({synthesis:null});assert.equal(missing.speaker.speak(),false);assert.equal(missing.clears(),0);
  const h=setup();h.speaker.speak();h.utterances[0].onerror();
  assert.equal(h.clears(),0);assert.equal(h.speaker.busy(),false);
  assert.match(h.statuses.at(-1),/문장은 유지/);
});

test('new input or a recording lock during speech startup is not cleared',()=>{
  for(const latest of [{raw:'새 문장',text:'새 문장입니다.'},{raw:'안녕하세요',text:'안녕하세요.',canClear:false}]){
    const h=setup();h.speaker.speak();h.setSnapshot(latest);h.utterances[0].onstart();
    assert.equal(h.clears(),0);h.utterances[0].onend();
  }
});

test('empty sentences, IME confirmation, repeat keys and form controls do not trigger speech',()=>{
  const h=setup();h.setSnapshot({raw:'대기 중',text:'대기 중'});assert.equal(h.speaker.speak(),false);
  assert.equal(isSpeechShortcut({key:'Enter'}),true);
  assert.equal(isSpeechShortcut({key:'Enter',target:{tagName:'BUTTON',closest:()=>null}}),true);
  for(const event of [{key:'a'},{key:'Enter',repeat:true},{key:'Enter',isComposing:true},{key:'Enter',keyCode:229},{key:'Enter',ctrlKey:true},
    {key:'Enter',target:{isContentEditable:true}},{key:'Enter',target:{closest:()=>({tagName:'INPUT'})}}])assert.equal(isSpeechShortcut(event),false);
});

test('a thrown synthesis error leaves text and the next sentence can be spoken',()=>{
  const h=setup({synthesis:{getVoices:()=>[],speak(){throw Error('unavailable');},cancel(){}}});
  assert.equal(h.speaker.speak(),false);assert.equal(h.clears(),0);assert.equal(h.speaker.busy(),false);
});

test('startup timeout and late events cannot erase current input',async()=>{
  const h=setup({startTimeoutMs:5});h.speaker.speak();
  await new Promise(resolve=>setTimeout(resolve,15));
  assert.equal(h.speaker.busy(),false);h.utterances[0].onstart();assert.equal(h.clears(),0);
});
