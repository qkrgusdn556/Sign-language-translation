import {classify, classifySequence, distance} from './recognition.js?v=tracking-recovery-1';

export const isJamo = label => /^[ㄱ-ㅎㅏ-ㅣ]$/u.test(label || '');

export function compareFingerspell(vector, sequence, samples, threshold=.8, {settled=false}={}) {
  samples=samples.filter(row=>row.label!=='__rest__');
  if (!vector) return {status:'unknown',similarity:0,reason:'tracking'};
  const candidates=vector.length===126?samples.map(row=>row.vector?
    {...row,vector:row.vector.slice(0,126)}:{...row,sequence:row.sequence?.map(frame=>frame.slice(0,126))}):samples;
  const compatible=candidates.filter(row=>row.vector?.length===vector.length ||
    row.sequence?.every(frame=>frame.length===vector.length));
  const counts=new Map();
  for(const row of compatible) if(isJamo(row.label)) counts.set(row.label,(counts.get(row.label)||0)+1);
  if(![...counts.values()].some(count=>count>=3)) return {status:'unknown',similarity:0,reason:'samples'};
  // One terminal observation per original recording, not repeated frames as extra samples.
  const poses=compatible.map(row=>row.vector?row:{label:row.label,vector:row.sequence.at(-1)});
  const pose=classify(vector,poses);
  const motion=sequence.length>=6?classifySequence(sequence,compatible.filter(row=>isJamo(row.label)&&row.sequence)):
    {status:'unknown',similarity:0};
  const strong=result=>result.status==='match' && result.similarity>=threshold;
  // Once the current shape has settled, a preceding letter's motion window
  // must not veto a confident static jamo. Wrist stillness alone is insufficient.
  const recent=sequence.slice(-4);
  const stableShape=settled && recent.length===4 && recent.every(frame=>distance(vector,frame)<=.08);
  if(stableShape && strong(pose) && isJamo(pose.label))
    return {...pose,source:'pose',settled:true};
  if(strong(pose)&&strong(motion)&&pose.label!==motion.label)
    return {status:'unknown',similarity:Math.max(pose.similarity,motion.similarity),reason:'conflict',poseLabel:pose.label,motionLabel:motion.label};
  const best=strong(pose)?{...pose,source:'pose'}:strong(motion)?{...motion,source:'motion'}:null;
  if(best) return best;
  return {status:'unknown',similarity:Math.max(pose.similarity||0,motion.similarity||0),
    reason:compatible.some(row=>isJamo(row.label))?'weak':'samples'};
}

export function fingerspellAgreement(yolo,joints,{yoloThreshold=.9,jointThreshold=.8}={}) {
  if(!isJamo(yolo.label)) return {accepted:false,reason:'other'};
  if(!Number.isFinite(yolo.confidence)||yolo.confidence<yoloThreshold) return {accepted:false,reason:'yolo-score'};
  if(joints.status!=='match'||!Number.isFinite(joints.similarity)||joints.similarity<jointThreshold)
    return {accepted:false,reason:joints.reason||'joint-score'};
  if(yolo.label!==joints.label) return {accepted:false,reason:'disagreement'};
  return {accepted:true,label:yolo.label};
}
