export function isSpeechShortcut(event,{allowRepeat=false}={}) {
  if(event.key!=='Enter'||(!allowRepeat&&event.repeat)||event.isComposing||event.keyCode===229||event.defaultPrevented||event.ctrlKey||event.altKey||event.metaKey||event.shiftKey)return false;
  const target=event.target;
  return !target?.isContentEditable && !target?.closest?.('input, textarea, select, a, [role="textbox"], [role="combobox"], [contenteditable]:not([contenteditable="false"])');
}

export function createSentenceSpeaker({synthesis,Utterance,getSnapshot,onStarted,onStatus=()=>{},onBusy=()=>{},startTimeoutMs=8000}) {
  let current=null,timer=null;
  function finish(message) {
    clearTimeout(timer);timer=null;current=null;onBusy(false);onStatus(message);
  }
  return {
    speak() {
      if(current)return false;
      const snapshot=getSnapshot();
      if(!snapshot?.raw||snapshot.raw==='대기 중'||!snapshot.text)return false;
      if(!synthesis||typeof Utterance!=='function') {onStatus('이 브라우저에서는 음성 읽기를 지원하지 않습니다.');return false;}
      const utterance=new Utterance(snapshot.text);
      utterance.lang='ko-KR';utterance.rate=1;utterance.pitch=1;
      const voice=synthesis.getVoices?.().find(v=>v.localService&&/^ko(?:-|_)/i.test(v.lang));
      if(voice)utterance.voice=voice;
      let started=false;
      current=utterance;onBusy(true);onStatus('음성 읽기 준비 중');
      utterance.onstart=()=>{
        if(current!==utterance)return;
        started=true;clearTimeout(timer);timer=null;
        const latest=getSnapshot();
        if(latest?.canClear!==false&&latest?.raw===snapshot.raw&&latest.text===snapshot.text) {
          onStarted(snapshot);onStatus('문장 읽는 중 · 입력 문장을 비웠습니다.');
        } else onStatus('문장 읽는 중 · 새로 입력된 문장은 유지합니다.');
      };
      utterance.onend=()=>{if(current===utterance)finish('문장 읽기 완료');};
      utterance.onerror=()=>{if(current===utterance)finish(started?'음성 읽기가 중단되었습니다.':'음성 읽기를 시작하지 못했습니다. 문장은 유지됩니다.');};
      timer=setTimeout(()=>{
        if(current!==utterance)return;
        finish('음성 읽기 시작 지연 · 문장은 유지됩니다.');synthesis.cancel();
      },startTimeoutMs);
      try{synthesis.speak(utterance);return true;}
      catch{finish('음성 읽기를 시작하지 못했습니다. 문장은 유지됩니다.');return false;}
    },
    busy:()=>Boolean(current),
    dispose(){if(current){finish('');synthesis?.cancel();}},
  };
}

export function installSentenceSpeech() {
  const raw=document.getElementById('sentenceText'),polished=document.getElementById('polishedSentenceText');
  const clear=document.getElementById('clearSentenceBtn'),button=document.getElementById('speakSentenceBtn'),status=document.getElementById('sentenceSpeechStatus');
  if(!raw||!polished||!clear||!button||!status)return;
  const snapshot=()=>{
    const original=raw.textContent.trim(),final=polished.textContent.trim();
    let text=final&&final!=='대기 중'?final:original;
    if(window.Hangul?.assemble)text=text.replace(/[ㄱ-ㅎㅏ-ㅣ]+(?:\s+[ㄱ-ㅎㅏ-ㅣ]+)*/gu,s=>window.Hangul.assemble([...s.replace(/\s/gu,'')]));
    return {raw:original,text,canClear:!clear.disabled};
  };
  const update=()=>{button.disabled=speaker.busy()||clear.disabled||raw.textContent.trim()==='대기 중'||!raw.textContent.trim();};
  const speaker=createSentenceSpeaker({synthesis:window.speechSynthesis,Utterance:window.SpeechSynthesisUtterance,getSnapshot:snapshot,
    onStarted:()=>clear.click(),onStatus:message=>{status.textContent=message;},onBusy:()=>update()});
  const speak=()=>{if(!clear.disabled)speaker.speak();};
  button.addEventListener('click',speak);
  const keydown=event=>{
    if(event.repeat&&speaker.busy()&&isSpeechShortcut(event,{allowRepeat:true})) {event.preventDefault();return;}
    if(!isSpeechShortcut(event)||document.querySelector('dialog[open]')||clear.disabled)return;
    if(snapshot().raw==='대기 중'||!snapshot().raw)return;
    event.preventDefault();speak();
  };
  document.addEventListener('keydown',keydown);
  const observer=new MutationObserver(update);
  observer.observe(raw,{childList:true,subtree:true,characterData:true});
  observer.observe(clear,{attributes:true,attributeFilter:['disabled']});
  update();
  window.addEventListener('pagehide',()=>{speaker.dispose();observer.disconnect();document.removeEventListener('keydown',keydown);},{once:true});
}

if(typeof document!=='undefined')installSentenceSpeech();
