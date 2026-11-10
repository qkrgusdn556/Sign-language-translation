export function assembleJamoText(raw, assemble) {
  return raw.replace(/[ㄱ-ㅎㅏ-ㅣ]+(?:\s+[ㄱ-ㅎㅏ-ㅣ]+)*/gu, text => assemble([...text.replace(/\s/gu, '')]));
}

export async function requestConfirmedSentence(body,signal,{fetchImpl=fetch,wait=ms=>new Promise(resolve=>setTimeout(resolve,ms)),onRetry=()=>{}}={}) {
  for(let attempt=0;attempt<2;attempt++) {
    const response=await fetchImpl('/api/gemini/polish',{
      method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body),signal});
    const value=await response.json();
    if(response.ok)return value;
    // Only retry an explicit upstream transient HTTP error, not local busy,
    // auth/quota errors, or a request that already timed out.
    if(!attempt && /HTTP (500|502|503|504)\b/u.test(value.error || '')) {
      onRetry();await wait(3200);continue;
    }
    throw Error(value.error || 'Gemini 요청 실패');
  }
}

export function createSentencePolisher({request, onResult, onStatus, delayMs=700, intervalMs=3200}) {
  let current=null, revision=0, timer=null, lastSent=-Infinity;
  let busy=false, activeGeneration=null, pending=null;
  function cancel() { clearTimeout(timer); pending=null; current=null; revision++; }
  function schedule() {
    clearTimeout(timer);
    if(busy || !pending) return;
    timer=setTimeout(()=>{
      const next=pending;pending=null;
      send(next.snapshot,next.generation);
    },Math.max(0,pending.readyAt-Date.now(),intervalMs-(Date.now()-lastSent)));
  }
  async function send(snapshot, generation) {
    if(generation!==revision) return;
    busy=true;activeGeneration=generation;
    const controller=new AbortController();
    lastSent=Date.now();
    onStatus('Gemini 문장 다듬는 중');
    try {
      const result=await request(snapshot,controller.signal);
      if(generation!==revision) return;
      onResult(result.sentence);
      onStatus('Gemini 다듬기 완료 · 원문 대조 필요');
    } catch(error) {
      if(generation===revision && error.name!=='AbortError') onStatus(error.message || 'Gemini 연결 실패');
    } finally {
      busy=false;activeGeneration=null;schedule();
    }
  }
  return {
    update(snapshot,{force=false}={}) {
      const next=JSON.stringify(snapshot);
      if(next===current) {
        if(!force || (busy && activeGeneration===revision)) return;
      } else {
        clearTimeout(timer);pending=null;current=next;revision++;
      }
      if(!snapshot.enabled || !snapshot.raw || snapshot.raw==='대기 중') return;
      const generation=revision;
      // Browser abort does not stop the server's Google request. Let it finish,
      // discard stale output, and keep only the latest input waiting for idle.
      pending={snapshot:{...snapshot},generation,readyAt:Date.now()+(force?0:delayMs)};
      onStatus(busy?'이전 요청 처리 중 · 최신 문장 자동 보정 대기':'입력 중 · 잠시 멈추면 자동 보정');
      schedule();
    },
    dispose:cancel,
  };
}

export function installSentencePolish() {
  const $=id=>document.getElementById(id);
  const enabled=$('geminiEnabled'), status=$('geminiStatus'), badge=$('privacyBadge');
  let configured=false,localKey=null;
  const request=async (path,body,signal)=>{
    const response=await fetch(`/api/gemini/${path}`,body===undefined?{cache:'no-store'}:{
      method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body),signal});
    const value=await response.json();
    if(!response.ok) throw Error(value.error || 'Gemini 요청 실패');
    return value;
  };
  const polisher=createSentencePolisher({
    async request(snapshot,signal) {
      return requestConfirmedSentence({raw:snapshot.raw,assembled:assembleJamoText(snapshot.raw,letters=>window.Hangul.assemble(letters)),
        style:snapshot.style,consent:true},signal,{onRetry:()=>{status.textContent='Gemini 일시적 서버 오류 · 한 번 다시 요청합니다.';}});
    },
    onResult:text=>{$('polishedSentenceText').textContent=assembleJamoText(text,letters=>window.Hangul.assemble(letters));},
    onStatus:text=>{status.textContent=text;},
  });
  function update(force=false) {
    const raw=$('sentenceText').textContent.trim(),style=$('casualStyle').checked?'casual':'polite';
    const key=JSON.stringify([raw,style]);
    if(key!==localKey || force) {
      localKey=key;
      $('polishedSentenceText').textContent=raw?assembleJamoText(raw,letters=>window.Hangul.assemble(letters)):'대기 중';
    }
    badge.textContent=enabled.checked?'영상·관절 로컬 처리 · 자동 보정 텍스트 Google 전송':'외부 전송 없음 · 로컬 처리';
    polisher.update({raw,style,
      enabled:enabled.checked&&configured},{force});
    if(enabled.checked&&!configured) status.textContent='Gemini API 키 연결 필요';
    if(!enabled.checked) status.textContent='한글 조합 즉시 표시 · AI 보정 전송 동의 꺼짐';
  }
  const observer=new MutationObserver(()=>update());
  observer.observe($('sentenceText'),{childList:true,characterData:true,subtree:true});
  enabled.addEventListener('change',()=>{
    update();
  });
  for(const id of ['politeStyle','casualStyle']) $(id).addEventListener('change',()=>update());
  $('polishSentenceBtn').addEventListener('click',()=>update(true));
  $('geminiConnect').addEventListener('click',async()=>{
    const key=$('geminiKey').value;
    $('geminiKey').value='';
    $('geminiConnect').disabled=true;
    try {
      const result=await request('configure',{...(key?{apiKey:key}:{}),model:$('geminiModel').value.trim()});
      configured=result.configured;update();
      status.textContent=`${result.model} 설정됨 · 자동 보정 준비`;
    } catch(error) {status.textContent=error.message;}
    finally {$('geminiConnect').disabled=false;}
  });
  request('status').then(result=>{
    configured=result.configured;$('geminiModel').value=result.model;
    status.textContent=configured?'API 키 설정됨 · Google 전송 동의 대기':'Gemini API 키 미설정';update();
  }).catch(()=>{status.textContent='Gemini 서버 연결 실패';});
  window.addEventListener('pagehide',()=>{observer.disconnect();polisher.dispose();});
}

if(typeof document!=='undefined') installSentencePolish();
