import test from 'node:test';
import assert from 'node:assert/strict';
import {createInputActivity} from '../frontend/input_activity.js';
const hand=(x=.5)=>Array.from({length:21},(_,i)=>({x:x+i*.003,y:.5-i*.003,z:0}));

test('fingerspelling recognizes a short 2.4 percent hand movement without changing word motion rules',()=>{
  const sign=createInputActivity(),jamo=createInputActivity({profile:'fingerspell'});
  let signMoved=false,jamoMoved=false;
  for(let i=0;i<20;i++) {
    const points=hand(.5+Math.min(i,4)*.006);
    signMoved ||= sign.update([points],i*75).moving;
    jamoMoved ||= jamo.update([points],i*75).moving;
  }
  assert.equal(signMoved,false);
  assert.equal(jamoMoved,true);
});

test('fingerspelling ignores stationary hands, alternating jitter and detection-order flicker',()=>{
  const gate=createInputActivity({profile:'fingerspell'});
  for(let i=0;i<120;i++) {
    const points=hand(.4+(i%2?.006:-.006));
    const other=hand(.8);
    for(const index of [4,8,12,16,20])points[index].x+=i%2?.002:-.002;
    points.forEach((p,j)=>{p.z=Math.sin(i+j)*2;});
    assert.equal(gate.update(i%2?[points,other]:[other,points],i*75).active,false);
  }
});

test('fingerspelling input resets after long tracking loss and mode reset',()=>{
  const gate=createInputActivity({profile:'fingerspell'});
  for(let i=0;i<8;i++)gate.update([hand(.4+i*.006)],i*75);
  gate.update([],700);gate.update([],1000);
  assert.equal(gate.update([hand(.7)],1100).active,false);
  gate.reset();
  assert.equal(gate.update([hand(.7)],1200).active,false);
});

test('stationary XY ignores depth noise and small detector jitter',()=>{
  const gate=createInputActivity();
  for(let i=0;i<120;i++) {
    const points=hand(.5+(i%2?.006:-.006));
    points.forEach((p,j)=>{p.z=Math.sin(i+j)*2;});
    assert.equal(gate.update([points],i*75).active,false);
  }
});

test('coherent movement arms input but expires after settling',()=>{
  const gate=createInputActivity();
  let moved=false;
  for(let i=0;i<12;i++) moved=gate.update([hand(.3+i*.02)],i*75).moving||moved;
  assert.equal(moved,true);
  for(let i=12;i<35;i++) gate.update([hand(.52)],i*75);
  assert.equal(gate.update([hand(.52)],35*75).active,false);
});

test('detection-order and hand-count flicker do not count as activity',()=>{
  const gate=createInputActivity();
  for(let i=0;i<80;i++) {
    const hands=i%2?[hand(.2),hand(.7)]:[hand(.7),hand(.2)];
    assert.equal(gate.update(hands,i*75).active,false);
  }
  for(let i=80;i<120;i++) assert.equal(gate.update(i%2?[hand(.7)]:[hand(.2),hand(.7)],i*75).active,false);
});

test('a single tracking jump and recovery cannot arm input',()=>{
  const gate=createInputActivity();
  for(let i=0;i<40;i++) assert.equal(gate.update([hand(i===15?.8:.5)],i*75).active,false);
});

test('brief hand loss pauses input without forcing a new motion, long loss clears activity',()=>{
  const gate=createInputActivity();
  for(let i=0;i<12;i++) gate.update([hand(.3+i*.02)],i*75);
  assert.equal(gate.update([],1000).active,false);
  assert.equal(gate.update([hand(.7)],1100).active,true);
  assert.equal(gate.update([],1200).active,false);
  assert.equal(gate.update([hand(.7)],1500).active,false);
  assert.equal(gate.update([hand(.3)],3000).active,false);
});

test('small continuous motion stays active after deliberate entry, but does not open from idle',()=>{
  const gate=createInputActivity();
  for(let i=0;i<12;i++) gate.update([hand(.2+i*.015)],i*75);
  for(let i=12;i<45;i++) assert.equal(gate.update([hand(.365+(i-11)*.0027)],i*75).active,true);
  gate.reset();
  for(let i=0;i<45;i++) assert.equal(gate.update([hand(.365+i*.0027)],i*75).active,false);
});

test('one active hand is not diluted by a resting second hand',()=>{
  const gate=createInputActivity();let moving=false;
  for(let i=0;i<20;i++) moving=gate.update([hand(.2+i*.0045),hand(.8)],i*75).moving||moving;
  assert.equal(moving,true);
});

test('intermittent loss of the other hand preserves the moving hand history',()=>{
  const gate=createInputActivity();let moving=false;
  for(let i=0;i<24;i++) {
    const hands=[hand(.2+i*.008)];
    if(i%2)hands.push(hand(.8));
    moving=gate.update(hands,i*75).moving||moving;
  }
  assert.equal(moving,true);
  for(let i=24;i<55;i++)gate.update([hand(.384),hand(.8)],i*75);
  assert.equal(gate.update([hand(.384)],55*75).active,false);
});

test('coherent finger motion can open input without moving the wrist across the image',()=>{
  const gate=createInputActivity();let moving=false;
  for(let i=0;i<12;i++) {
    const points=hand();
    for(const index of [4,8,12,16,20])points[index].x+=i*.004;
    moving=gate.update([points],i*75).moving||moving;
  }
  assert.equal(moving,true);
});

test('small finger-only jitter cannot open input',()=>{
  const gate=createInputActivity();
  for(let i=0;i<80;i++) {
    const points=hand();
    for(const index of [4,8,12,16,20])points[index].x+=i%2?.002:-.002;
    assert.equal(gate.update([points],i*75).active,false);
  }
});

test('intermediate camera frames retain the last measured movement decision',()=>{
  const gate=createInputActivity();
  let result;
  for(let i=0;i<12;i++)result=gate.update([hand(.2+i*.02)],i*90);
  assert.equal(result.moving,true);
  assert.equal(gate.update([hand(.42)],11*90+45).moving,true);
  for(let i=12;i<30;i++)result=gate.update([hand(.42)],i*90);
  assert.equal(result.moving,false);
  assert.equal(gate.update([hand(.42)],29*90+45).moving,false);
});
