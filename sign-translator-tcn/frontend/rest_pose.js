import {distance,trackedHandCount,classifySequence} from './recognition.js?v=tracking-recovery-1';

export const REST_LABEL='__rest__';
export const isRestLabel=label=>label===REST_LABEL;
const valid=vector=>Array.isArray(vector)&&vector.length===130&&vector.every(Number.isFinite)&&trackedHandCount(vector)>0;

function wristDistance(first,second) {
  const count=trackedHandCount(first);
  if(count===1) {
    const a=first.slice(0,63).some(n=>n!==0)?0:1,b=second.slice(0,63).some(n=>n!==0)?0:1;
    return Math.hypot(first[126+a*2]-second[126+b*2],first[127+a*2]-second[127+b*2]);
  }
  const difference=swap=>{
    const values=[];
    for(let hand=0;hand<count;hand++) {
      const other=swap?1-hand:hand;
      values.push(Math.hypot(first[126+hand*2]-second[126+other*2],first[127+hand*2]-second[127+other*2]));
    }
    return Math.max(...values);
  };
  return Math.min(difference(false),count===2?difference(true):Infinity);
}

export function compareRestPose(vector,samples) {
  const poses=samples.filter(row=>isRestLabel(row.label)).map(row=>row.vector||row.sequence?.at(-1)).filter(valid);
  if(!valid(vector))return {matched:false,similarity:0,count:poses.length,reason:'tracking'};
  const compatible=poses.filter(pose=>trackedHandCount(pose)===trackedHandCount(vector));
  const nearby=compatible.filter(pose=>wristDistance(vector,pose)<=.12).map(pose=>distance(vector,pose)).sort((a,b)=>a-b);
  const similarity=nearby.length>=2?Math.exp(-(nearby[0]+nearby[1])/2/.2):0;
  const result={matched:compatible.length>=3&&nearby.length>=2&&similarity>=.75,similarity,count:poses.length,
    hands:trackedHandCount(vector),compatibleCount:compatible.length,nearbyCount:nearby.length,
    reason:compatible.length<3?'samples':nearby.length<2?'position':similarity>=.75?'match':'shape'};
  if(result.matched)return result;
  // Try observed parts of paired references even when single-hand examples already exist.
  {
    // Missing-hand evidence can veto output, but never confirm a complete rest pose.
    if(trackedHandCount(vector)===1) {
      const visible=vector.slice(0,63).some(n=>n!==0)?0:1;
      const references=poses.filter(pose=>trackedHandCount(pose)===2).map(pose=>{
        const choices=[0,1].map(slot=>{
          const projected=Array(130).fill(0);
          projected.splice(visible*63,63,...pose.slice(slot*63,slot*63+63));
          projected[126+visible*2]=pose[126+slot*2];projected[127+visible*2]=pose[127+slot*2];
          return projected;
        });
        choices.sort((a,b)=>distance(vector,a)-distance(vector,b));
        return {label:REST_LABEL,vector:choices[0]};
      });
      if(references.length>=3) {
        const partial=compareRestPose(vector,references);
        if(partial.matched)return {...result,similarity:partial.similarity,matched:false,partialMatch:true,reason:'partial'};
      }
    }
  }
  return result;
}

export function createRestPoseGate() {
  let active=false,since=null,lastMatch=-Infinity,lastAt=-Infinity,version=null,frames=[],lastSample=-Infinity,lastCompare=-Infinity,dynamic=null;
  return {
    reset(){active=false;since=null;lastMatch=lastAt=-Infinity;version=null;frames=[];lastSample=lastCompare=-Infinity;dynamic=null;},
    update(vector,samples,now,{moving=false}={}) {
      if(version!==samples||now-lastAt>300||now<lastAt){this.reset();version=samples;}
      lastAt=now;
      const result=compareRestPose(vector,samples),wasActive=active;
      if(valid(vector)) {
        if(frames.length&&trackedHandCount(frames.at(-1))!==trackedHandCount(vector)){frames=[];dynamic=null;}
        if(now-lastSample>=75){frames.push([...vector]);if(frames.length>20)frames.shift();lastSample=now;}
        const motions=samples.filter(row=>row.sequence?.length>=6&&!/^[ㄱ-ㅎㅏ-ㅣ]$/u.test(row.label));
        const restCount=motions.filter(row=>isRestLabel(row.label)).length;
        if(restCount>=3&&frames.length>=6&&now-lastCompare>=150){dynamic=classifySequence(frames,motions);lastCompare=now;}
        result.sequenceSimilarity=dynamic?.similarity||0;
        result.sequenceLabel=dynamic?.label||dynamic?.nearestLabel||null;
        // Use the same DTW decision as words. A final-frame gate used to reject good clips.
        if(!moving&&dynamic?.status==='match') {
          if(isRestLabel(dynamic.label)) {
            result.matched=!dynamic.partial;result.partialMatch=Boolean(dynamic.partial);
            result.similarity=dynamic.similarity;result.reason=dynamic.partial?'partial':'dtw';
          } else {
            result.matched=false;result.partialMatch=false;result.reason='other-sign';
            result.otherLabel=dynamic.label;
          }
        }
      }else{frames=[];dynamic=null;}
      if(result.matched) {
        lastMatch=now;
        if(since===null)since=now;
        if(!active&&!moving&&now-since>=180)active=true;
      } else {
        since=null;
        // Real departure releases immediately; a single quiet detector wobble gets a short grace.
        if(moving||result.reason==='tracking'||result.reason==='samples'||now-lastMatch>120)active=false;
      }
      return {...result,active,blocked:active||result.matched||Boolean(result.partialMatch),entered:!wasActive&&active,left:wasActive&&!active};
    },
  };
}
