const valid=hand=>hand?.length===21&&hand.every(p=>[p.x,p.y,p.z].every(Number.isFinite));
const distance=(a,b)=>Math.hypot(a.x-b.x,a.y-b.y);

export function handCrop(hand,width,height) {
  if(!valid(hand)||width<=0||height<=0)return null;
  const xs=hand.map(p=>p.x*width),ys=hand.map(p=>p.y*height);
  const left=Math.min(...xs),right=Math.max(...xs),top=Math.min(...ys),bottom=Math.max(...ys);
  const extent=Math.max(right-left,bottom-top);
  if(extent<12||extent>220)return null;
  const size=Math.round(Math.max(96,extent*1.7));
  if(size>Math.min(width,height)*.8)return null;
  const x=Math.round((left+right-size)/2),y=Math.round((top+bottom-size)/2);
  if(x<0||y<0||x+size>width||y+size>height)return null;
  return {x,y,size,width,height};
}

export function mapCropHand(hand,crop) {
  if(!valid(hand))return null;
  return hand.map(p=>({x:(crop.x+p.x*crop.size)/crop.width,y:(crop.y+p.y*crop.size)/crop.height,z:p.z*crop.size/crop.width}));
}

export function acceptRefinement(original,refined) {
  if(!valid(original)||!valid(refined)||refined.some(p=>p.x<0||p.x>1||p.y<0||p.y>1))return false;
  const scale=Math.max(.02,distance(original[0],original[9]));
  if(distance(original[0],refined[0])>Math.max(.012,scale*.4))return false;
  const extent=distance(refined[0],refined[9])/scale;
  if(extent<.65||extent>1.5)return false;
  return original.reduce((sum,p,i)=>sum+distance(p,refined[i]),0)/21<scale*.35;
}

export function createHandPrecision({onStatus=()=>{},clock=()=>performance.now(),
  makeWorker=()=>new Worker(new URL('./hand_precision_worker.js',import.meta.url)),
  bitmap=(video,crop)=>createImageBitmap(video,crop.x,crop.y,crop.size,crop.size,{resizeWidth:512,resizeHeight:512}),
  supported=()=>typeof Worker!=='undefined'&&typeof createImageBitmap==='function',timeoutMs=35}={}) {
  let worker=null,ready=false,pending=null,inFlight=null,serial=0,lastTry=-Infinity,blockedUntil=0,nextHand=0,startTimer=null;
  function settle(value) {
    if(!pending)return;
    clearTimeout(pending.timer);const resolve=pending.resolve;pending=null;resolve(value);
  }
  const api={
    ready:()=>ready&&!pending&&inFlight===null,
    start() {
      if(worker||!supported())return;
      try {
        worker=makeWorker();const current=worker;
        onStatus('확대 모델 준비');
        startTimer=setTimeout(()=>{if(worker===current&&!ready){api.stop();onStatus('기본 추적 · 확대 모델 지연');}},15000);
        worker.onmessage=({data})=>{
          if(worker!==current)return;
          if(data.type==='ready'){clearTimeout(startTimer);ready=true;onStatus('손 정밀 추적');return;}
          if(data.type==='error'){console.warn('Hand precision worker:',data.message||'initialization failed');api.stop();onStatus('기본 추적 · 확대 모델 불가');return;}
          if(data.id===inFlight)inFlight=null;
          if(data.id!==pending?.id)return;
          settle(data.result||null);
        };
        worker.onerror=event=>{if(worker===current){console.warn('Hand precision worker:',event.message||'worker failed');api.stop();onStatus('기본 추적 · 확대 모델 불가');}};
      }catch{api.stop();onStatus('기본 추적 · 확대 모델 불가');}
    },
    stop() {
      ready=false;clearTimeout(startTimer);settle(null);worker?.terminate();worker=null;
      serial++;lastTry=-Infinity;blockedUntil=0;nextHand=0;inFlight=null;
    },
    async refine(raw,video,now,baseMs=0) {
      if(!ready||pending||inFlight!==null||now-lastTry<250||clock()<blockedUntil||baseMs>20)return {result:raw,observation:null};
      const hands=raw.landmarks||[],width=video.videoWidth,height=video.videoHeight;
      let selected=null;
      for(let n=0;n<hands.length;n++) {
        const index=(nextHand+n)%hands.length,crop=handCrop(hands[index],width,height);
        if(!crop)continue;
        // Do not zoom into overlapping hands: ownership would be ambiguous.
        const overlaps=hands.some((hand,j)=>j!==index&&hand.some(p=>p.x*width>=crop.x&&p.x*width<=crop.x+crop.size&&p.y*height>=crop.y&&p.y*height<=crop.y+crop.size));
        if(!overlaps){selected={index,crop};break;}
      }
      if(!selected)return {result:raw,observation:null};
      lastTry=now;nextHand=selected.index+1;
      const current=worker,id=++serial,started=clock();
      let expired=false;
      const response=new Promise(resolve=>{
        const timer=setTimeout(()=>{expired=true;blockedUntil=clock()+1000;settle(null);onStatus('기본 추적 · 확대 속도 대기');},timeoutMs);
        pending={id,resolve,timer};
      });
      // Include bitmap creation in the deadline and close late bitmaps without using them.
      Promise.resolve().then(()=>bitmap(video,selected.crop)).then(image=>{
        if(worker!==current||pending?.id!==id){image.close();return;}
        try{inFlight=id;worker.postMessage({id,image},[image]);}catch{inFlight=null;image.close();settle(null);}
      }).catch(()=>settle(null));
      const detected=await response,elapsed=clock()-started;
      const source=detected?.landmarks?.[0],refined=mapCropHand(source,selected.crop);
      const category=detected?.handedness?.[0]?.[0],originalCategory=raw.handedness?.[selected.index]?.[0];
      const conflict=category?.score>=.8&&originalCategory?.score>=.8&&category.categoryName!==originalCategory.categoryName;
      const accepted=!expired&&elapsed<=timeoutMs&&!conflict&&detected?.landmarks?.length===1&&acceptRefinement(hands[selected.index],refined);
      const observation={index:selected.index,crop:selected.crop,elapsed_ms:elapsed,accepted,
        reason:expired?'deadline':accepted?'agreement':conflict?'handedness':!source?'not-detected':'geometry',landmarks:refined||null};
      if(!accepted)return {result:raw,observation};
      const landmarks=hands.map((hand,i)=>i===selected.index?refined:hand);
      const worldLandmarks=hands.map((_,i)=>i===selected.index?detected.worldLandmarks?.[0]||null:raw.worldLandmarks?.[i]||null);
      onStatus('손 정밀 추적');
      return {result:{...raw,landmarks,worldLandmarks},observation};
    },
  };
  return api;
}
