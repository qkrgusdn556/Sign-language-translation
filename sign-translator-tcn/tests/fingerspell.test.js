import test from "node:test";
import assert from "node:assert/strict";
import vm from "node:vm";
import { readFileSync } from "node:fs";
import { createFingerspelling, createAutomaticFingerspelling } from "../frontend/fingerspell.js";
const context = { window: {} };
vm.runInNewContext(readFileSync(new URL("../vendor/hangul.js", import.meta.url), "utf8"), context);
const create = () => createFingerspelling(letters => context.window.Hangul.assemble(letters));

test('explicit YOLO-only observations use hold time without requiring joint motion',()=>{
  const s=createAutomaticFingerspelling(letters=>letters.join(''));
  for(const t of [0,150,300])assert.equal(s.observe('ㅎ',.65,t,{threshold:.6,requireMotion:false}),false);
  assert.equal(s.observe('ㅎ',.65,450,{threshold:.6,requireMotion:false}),true);
  for(let t=600;t<4000;t+=150)assert.equal(s.observe('ㅎ',.65,t,{threshold:.6,requireMotion:false}),false);
  assert.equal(s.text(),'ㅎ');
  s.observe('ㅏ',.7,4000,{threshold:.6,requireMotion:false});
  assert.equal(s.observe('ㅏ',.7,4300,{threshold:.6,requireMotion:false}),false);
  assert.equal(s.observe('ㅏ',.7,4450,{threshold:.6,requireMotion:false}),true);
  assert.equal(s.text(),'ㅎㅏ');
});

test("70 percent threshold reaches automatic composition without bypassing hold or duplicate guards", () => {
  const s = createAutomaticFingerspelling(letters => letters.join(''));
  s.noteMotion(0);
  assert.equal(s.observe('ㄱ', .69, 0, {threshold:.7}), false);
  assert.equal(s.observe('ㄱ', .7, 100, {threshold:.7}), false);
  assert.equal(s.observe('ㄱ', .7, 400, {threshold:.7}), false);
  assert.equal(s.observe('ㄱ', .7, 550, {threshold:.7}), true);
  assert.equal(s.text(), 'ㄱ');
  assert.equal(s.observe('ㄱ', .99, 700, {threshold:.7}), false);
  const manual = create();
  manual.observe('ㄱ', .7, 0, {threshold:.7});
  assert.equal(manual.observe('ㄱ', .7, 300, {threshold:.7}), true);
  const defaults = create();
  defaults.observe('ㄱ', .7, 0);
  assert.equal(defaults.observe('ㄱ', .7, 300), false);
});

test("assembles proper names, compound vowels and final consonants", () => {
  const s = create();
  let now = 0;
  for (const jamo of "ㅂㅏㄱㅎㅕㄴㅇㅜ") {
    s.observe(jamo, .99, now);
    assert.equal(s.observe(jamo, .99, now + 300), true);
    now += 500;
  }
  assert.equal(s.text(), "박현우");
  s.backspace();
  assert.equal(s.text(), "박현ㅇ");
  assert.equal(context.window.Hangul.assemble([..."ㄱㅗㅏㄴ"]), "관");
});

test("held letters do not repeat, explicit release allows repeats, invalid candidates do not commit", () => {
  const s = create();
  s.observe("ㄱ",.99,0);
  s.observe("ㄱ",.99,300);
  assert.equal(s.observe("ㄱ",.99,900),false);
  s.release();
  s.observe("ㄱ",.99,1000);
  assert.equal(s.observe("ㄱ",.99,1300),true);
  s.observe("ㅏ",.99,1400);
  s.resetCandidate();
  assert.equal(s.observe("ㅏ",.99,3000),false);
  assert.equal(s.observe("ㅏ",.1,4000),false);
  assert.equal(s.observe("unknown",.99,5000),false);
  assert.equal(s.letters().length,2);
});

test("automatic letters update one segment, block overlapping words, and expire without a finish button", () => {
  const texts=[], ended=[];
  const s=createAutomaticFingerspelling(letters=>context.window.Hangul.assemble(letters),{onText:t=>texts.push(t),onEnd:t=>ended.push(t)});
  s.noteMotion(0);
  for(const time of [0,150,300,450,600]) s.observe('ㄱ',.99,time);
  assert.deepEqual(texts,['ㄱ']);
  assert.equal(s.blocksWord(700),true);
  s.noteMotion(750);
  for(const time of [750,900,1050,1200]) s.observe('ㄴ',.99,time);
  assert.deepEqual(texts,['ㄱ','ㄱㄴ']);
  assert.equal(s.blocksWord(1800),false);
  s.tick(2700);
  assert.deepEqual(ended,['ㄱㄴ']);
  assert.equal(s.text(),'');
});

test("automatic repetition needs sustained confident other, not hand loss or weak flicker", () => {
  const texts=[];
  const s=createAutomaticFingerspelling(letters=>letters.join(''),{onText:t=>texts.push(t)});
  s.noteMotion(0);
  for(const time of [0,150,300,450]) s.observe('ㄱ',.99,time);
  s.pause();
  for(const time of [600,750,900,1050]) s.observe('ㄱ',.99,time);
  assert.equal(texts.length,1);
  for(const time of [1100,1250,1400,1550]) s.observe('기타',.99,time);
  s.noteMotion(1600);
  for(const time of [1600,1750,1900,2050]) s.observe('ㄱ',.99,time);
  assert.deepEqual(texts,['ㄱ','ㄱㄱ']);
  assert.equal(s.observe('ㅏ',.89,2200),false);
});

test("a stale inference gap cannot satisfy automatic hold time", () => {
  const s=createAutomaticFingerspelling(letters=>letters.join(''));
  s.noteMotion(0);
  assert.equal(s.observe('ㄱ',.99,0),false);
  assert.equal(s.observe('ㄱ',.99,800),false);
  assert.equal(s.text(),'');
});

test("idle high confidence, label flicker and segment timeout cannot manufacture letters", () => {
  const texts=[];
  const s=createAutomaticFingerspelling(letters=>letters.join(''),{onText:t=>texts.push(t)});
  for(let t=0;t<4000;t+=150) s.observe('ㄴ',1,t);
  assert.deepEqual(texts,[]);
  s.noteMotion(4000);
  for(let t=4000;t<=4600;t+=150) s.observe('ㄱ',1,t);
  for(let t=4750;t<10000;t+=150) s.observe(t%1200<600?'ㄴ':'ㄱ',1,t);
  s.tick(12000);
  for(let t=12000;t<14000;t+=150) s.observe('ㄴ',1,t);
  assert.deepEqual(texts,['ㄱ']);
});

test('joint-only threshold is explicit and does not change the default YOLO threshold',()=>{
  const joint=createAutomaticFingerspelling(letters=>letters.join(''));joint.noteMotion(0);
  for(const time of [0,150,300,450])joint.observe('ㄱ',.85,time,{threshold:.8});
  assert.equal(joint.text(),'ㄱ');
  const defaultMode=createAutomaticFingerspelling(letters=>letters.join(''));defaultMode.noteMotion(0);
  for(const time of [0,150,300,450])defaultMode.observe('ㄱ',.85,time);
  assert.equal(defaultMode.text(),'');
});
