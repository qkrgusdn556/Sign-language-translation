const PALM = [0, 5, 9, 13, 17];
const xy = (a,b) => Math.hypot(a.x-b.x,a.y-b.y);
const scale = hand => Math.max(.015,xy(hand[0],hand[9]));
const valid = hand => hand?.length===21 && hand.every(p=>[p.x,p.y,p.z].every(Number.isFinite));
const copy = hand => hand.map(p=>({...p}));
const shapeDistance = (a,b) => a.reduce((sum,p,i)=>sum+Math.hypot(
  (p.x-a[0].x)/scale(a)-(b[i].x-b[0].x)/scale(b),
  (p.y-a[0].y)/scale(a)-(b[i].y-b[0].y)/scale(b)),0)/21;

function duplicate(a,b) {
  const size=Math.min(scale(a),scale(b));
  const ratio=scale(a)/scale(b);
  // Finger estimates can disagree while both detections describe the same palm.
  // Require coincident wrists and all palm anchors, not coincident fingertips.
  return ratio>=.65 && ratio<=1/.65 && xy(a[0],b[0]) < size*.35 &&
    PALM.every(i=>xy(a[i],b[i])<size*.4);
}

export function createHandTracker() {
  let tracks=[null,null], lastAt=-Infinity;
  return {
    reset() { tracks=[null,null];lastAt=-Infinity; },
    update(result,now) {
      if(now-lastAt>250) this.reset();
      lastAt=now;
      tracks=tracks.map(track=>track&&now-track.at<=220?track:null);
      let entries=(result.landmarks||[]).slice(0,2).map((hand,index)=>({hand,index})).filter(item=>valid(item.hand));
      let duplicates=0, suppressed=false;
      if(entries.length===2 && duplicate(entries[0].hand,entries[1].hand)) {
        const cost=hand=>Math.min(...tracks.filter(Boolean).map(old=>xy(old.points[0],hand[0])/scale(old.points)+shapeDistance(old.points,hand)));
        if(tracks.some(Boolean)) entries.sort((a,b)=>cost(a.hand)-cost(b.hand));
        else entries.sort((a,b)=>(result.handedness?.[b.index]?.[0]?.score||0)-(result.handedness?.[a.index]?.[0]?.score||0));
        entries=entries.slice(0,1);duplicates=1;
      }
      const cost=({hand,index},slot)=>{
        const track=tracks[slot];
        if(!track)return 4;
        const dt=Math.min(.12,Math.max(0,(now-track.at)/1000));
        const predicted={x:track.raw[0].x+track.velocity.x*dt,y:track.raw[0].y+track.velocity.y*dt};
        const category=result.handedness?.[index]?.[0];
        const mismatch=category?.score>=.8&&track.category?.score>=.8&&category.categoryName!==track.category.categoryName;
        return xy(predicted,hand[0])/Math.max(.04,scale(track.points))+Math.min(2,shapeDistance(track.points,hand))*.35+(mismatch ? .7 : 0);
      };
      // Compare both assignments together; a greedy first hand can steal the second's track.
      let assignment=entries.length===2
        ? cost(entries[0],0)+cost(entries[1],1)<=cost(entries[0],1)+cost(entries[1],0)?[0,1]:[1,0]
        : entries.length?[cost(entries[0],0)<=cost(entries[0],1)?0:1]:[];
      const filtered=entries.map(({hand,index},entry)=>{
        const slot=assignment[entry];
        let track=tracks[slot];
        if(track&&xy(track.raw[0],hand[0])>.28)track=null;
        const old=track?.points;
        let points=copy(hand), held=false, alpha=1;
        if(old) {
          // A sudden shape change at a nearly stationary wrist needs a second observation.
          const abrupt=xy(old[0],hand[0])<scale(old)*.5 && shapeDistance(old,hand)>.65;
          if(abrupt && (!track.pending || shapeDistance(track.pending,hand)>.25)) {
            points=copy(old);suppressed=held=true;
          } else {
            const displacement=Math.max(xy(old[0],hand[0]),...hand.map((p,i)=>xy(p,old[i])));
            const base=Math.min(.94,.18+displacement/scale(hand)*2);
            alpha=1-Math.pow(1-base,Math.min(2,Math.max(.3,(now-track.at)/50)));
            points=hand.map((p,i)=>({x:old[i].x+alpha*(p.x-old[i].x),
              y:old[i].y+alpha*(p.y-old[i].y),z:old[i].z+alpha*(p.z-old[i].z)}));
          }
        }
        const rawWorld=result.worldLandmarks?.[index], oldWorld=track?.world;
        let world;
        if(held && oldWorld) world=copy(oldWorld);
        else if(!held && valid(rawWorld)) world=oldWorld?rawWorld.map((p,i)=>({
          x:oldWorld[i].x+alpha*(p.x-oldWorld[i].x),y:oldWorld[i].y+alpha*(p.y-oldWorld[i].y),
          z:oldWorld[i].z+alpha*(p.z-oldWorld[i].z)})):copy(rawWorld);
        const dt=track?Math.max(.015,(now-track.at)/1000):1;
        const velocity=track&&!held?{x:Math.max(-2,Math.min(2,(hand[0].x-track.raw[0].x)/dt)),y:Math.max(-2,Math.min(2,(hand[0].y-track.raw[0].y)/dt))}:track?.velocity||{x:0,y:0};
        const category=track?.category?.score>=.65?track.category:result.handedness?.[index]?.[0];
        tracks[slot]={points,raw:copy(hand),world:world||null,velocity,category,pending:held?copy(hand):null,at:now};
        return {points,index,world,slot,category};
      });
      filtered.sort((a,b)=>a.slot-b.slot);
      return {...result,
        landmarks:filtered.map(item=>item.points),
        handedness:filtered.map(item=>item.category?[item.category]:[]),
        worldLandmarks:filtered.map(item=>item.world),
        tracking:{duplicates,suppressed,handIds:filtered.map(item=>item.slot)},
      };
    },
  };
}
