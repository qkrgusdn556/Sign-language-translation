export function createRawCapture({onState=()=>{},onSaved=()=>{},now=()=>performance.now(),makeRecorder=(stream,options)=>new MediaRecorder(stream,options),post=async body=>{
  const response=await fetch('/api/recordings',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
  const value=await response.json();if(!response.ok)throw Error(value.error||'영상 저장 실패');return value;
}}={}) {
  let phase='idle', recorder=null, chunks=[], metadata=null, began=0, timer=null, size=0, pending=null;
  const state=message=>onState({phase,message});
  function fail(error){phase='error';state(error.message||String(error));}
  function encoded(blob) {
    return new Promise((resolve,reject)=>{const reader=new FileReader();reader.onload=()=>resolve(reader.result.split(',')[1]);reader.onerror=()=>reject(Error('영상 읽기 실패'));reader.readAsDataURL(blob);});
  }
  async function save() {
    phase='saving';state('컴퓨터에 저장 중');
    try {const value=await post(pending);pending=null;phase='idle';state('저장 완료 · 모델 학습 미반영');onSaved(value.clip,value);}
    catch(error){fail(error);}
  }
  return {
    active:()=>phase==='recording',
    busy:()=>['recording','finalizing','saving'].includes(phase),
    phase:()=>phase,
    start(stream,details) {
      if(this.busy()||pending)throw Error('진행 중이거나 저장하지 못한 영상이 있습니다.');
      if(!stream||!stream.getVideoTracks().some(track=>track.readyState==='live'))throw Error('카메라를 먼저 켜 주세요.');
      const mime=['video/webm;codecs=vp9','video/webm;codecs=vp8','video/webm'].find(type=>MediaRecorder.isTypeSupported(type));
      if(!mime)throw Error('이 브라우저는 WebM 녹화를 지원하지 않습니다.');
      const videoOnly=new MediaStream(stream.getVideoTracks());
      recorder=makeRecorder(videoOnly,{mimeType:mime,videoBitsPerSecond:1800000});
      metadata={...details,schema_version:1,request_id:crypto.randomUUID().replaceAll('-',''),mime_type:mime,
        started_at_utc:new Date().toISOString(),audio:false,frames:[],predictions:[]};
      chunks=[];size=0;began=now();
      recorder.ondataavailable=event=>{if(event.data.size){chunks.push(event.data);size+=event.data.size;if(size>32*1024*1024)this.stop();}};
      recorder.onerror=event=>{clearTimeout(timer);pending=null;chunks=[];fail(event.error||Error('영상 녹화 실패'));};
      recorder.onstop=async()=>{
        clearTimeout(timer);
        if(phase==='error')return;
        try {
          if(size>32*1024*1024)throw Error('영상이 32MB를 초과했습니다. 다시 촬영해 주세요.');
          const blob=new Blob(chunks,{type:mime});chunks=[];
          if(!blob.size)throw Error('영상 데이터가 없습니다. 다시 촬영해 주세요.');
          pending={metadata,video_base64:await encoded(blob)};
          await save();
        }catch(error){fail(error);}
      };
      recorder.start(250);phase='recording';state('촬영 중');
      timer=setTimeout(()=>this.stop(),30000);
    },
    observe({now:at,...frame}) {
      if(phase!=='recording')return;
      if(metadata.frames.length<1000)metadata.frames.push({...frame,elapsed_ms:Math.max(0,at-began)});
    },
    prediction(label,source,key=null) {
      if(phase!=='recording')return;
      const event={label,source,at_ms:Math.max(0,now()-began)};
      const index=key===null?-1:metadata.predictions.findIndex(row=>row.key===key);
      if(index>=0)metadata.predictions.splice(index,1);
      if(metadata.predictions.length<500)metadata.predictions.push(key===null?event:{...event,key});
    },
    stop() {
      if(phase!=='recording')return;
      metadata.duration_ms=Math.max(0,now()-began);phase='finalizing';clearTimeout(timer);state('촬영 종료 중');
      if(recorder.state!=='inactive')recorder.stop();
    },
    cancel() {
      if(phase!=='recording')return;
      clearTimeout(timer);recorder.ondataavailable=null;recorder.onstop=null;
      if(recorder.state!=='inactive')recorder.stop();
      pending=null;chunks=[];metadata=null;phase='idle';state('촬영 취소');
    },
    async retry(){if(!pending||phase!=='error')return;await save();},
    discard(){if(this.busy())return;pending=null;chunks=[];phase='idle';state('대기');},
  };
}

export function installCapturePanel({getStream,getPipeline,getModelIdentity=async()=>({}),onBegin=()=>{},onChange=()=>{},onSaved:afterSaved=()=>{}}) {
  const element=id=>document.getElementById(id);
  const ids=['captureSession','captureSplit','captureKind','captureSequence','captureSegments','captureConsent','captureNewSession','captureShareSlrt'];
  let context=null,lockedSession=null,externalBusy=false;
  const capture=createRawCapture({
    onState({phase,message}) {
      element('rawCaptureStatus').textContent=message;
      element('rawCaptureStart').disabled=externalBusy||!['idle'].includes(phase);
      element('rawCaptureStop').disabled=phase!=='recording';
      element('rawCaptureRetry').hidden=phase!=='error';element('rawCaptureDiscard').hidden=phase!=='error';
      for(const id of ids)element(id).disabled=externalBusy||['recording','finalizing','saving'].includes(phase);
      if(lockedSession===element('captureSession').value)element('captureSplit').disabled=true;
      if(['rest','non_sign','transition'].includes(element('captureKind').value)) {
        element('captureSequence').disabled=true;element('captureSegments').disabled=true;
      }
      if(element('captureKind').value==='continuous')element('captureShareSlrt').disabled=true;
      onChange();
    },
    onSaved(clip,response) {lockedSession=clip.session_id;element('captureSplit').disabled=true;
      if(response?.slrt)element('rawCaptureStatus').textContent=response.slrt.status==='error'?`저장 완료 · SLRT 전달 실패: ${response.slrt.error}`:'저장 완료 · SLRT 학습 자료 전달됨 · 학습 대기';
      element('rawCapturePlayback').src=`/api/recordings/${clip.clip_id}/video`;element('rawCapturePlayback').hidden=false;
      refresh().catch(showError);
      afterSaved(clip,response);
    },
  });
  capture.setExternalBusy=busy=>{
    externalBusy=Boolean(busy);
    for(const id of ids)element(id).disabled=externalBusy||capture.busy();
    element('rawCaptureStart').disabled=externalBusy||capture.phase()!=='idle';
    if(lockedSession===element('captureSession').value)element('captureSplit').disabled=true;
    if(['rest','non_sign','transition'].includes(element('captureKind').value)) {
      element('captureSequence').disabled=true;element('captureSegments').disabled=true;
    }
    if(element('captureKind').value==='continuous')element('captureShareSlrt').disabled=true;
  };
  capture.noteSharedSession=session=>{lockedSession=session;if(element('captureSession').value===session)element('captureSplit').disabled=true;};
  function showError(error){element('rawCaptureStatus').textContent=error.message||String(error);}
  async function refresh() {
    const response=await fetch('/api/recordings');if(!response.ok)throw Error('수집 API 연결 필요 · 서버를 다시 시작해 주세요.');
    const {clips}=await response.json();
    element('captureList').replaceChildren();
    for(const clip of clips.slice(-10).reverse()) {
      const split={train:'학습',validation:'검증',test:'최종 평가'}[clip.split],kind={isolated:'한 단어',continuous:'연속 수어',rest:'휴식',non_sign:'일상 행동',transition:'전환 동작'}[clip.kind];
      const row=document.createElement('li');row.textContent=`${split} · ${kind} · ${clip.sequence.join(' → ')||'비수어'} · ${(clip.duration_ms/1000).toFixed(1)}초 · ${clip.session_id}`;
      element('captureList').append(row);
    }
    element('captureCount').textContent=`${clips.length}개`;
  }
  element('captureSession').value=crypto.randomUUID();
  element('captureNewSession').addEventListener('click',()=>{element('captureSession').value=crypto.randomUUID();lockedSession=null;element('captureSplit').disabled=false;});
  element('captureKind').addEventListener('change',()=>{
    element('captureSequence').disabled=['rest','non_sign','transition'].includes(element('captureKind').value);
    element('captureSegments').disabled=element('captureSequence').disabled;
    element('captureShareSlrt').disabled=element('captureKind').value==='continuous';
  });
  element('rawCaptureStart').addEventListener('click',async()=>{
    if(capture.busy())return;
    element('rawCaptureStart').disabled=true;
    try {
      if(!element('captureConsent').checked)throw Error('영상·관절의 컴퓨터 저장에 동의해 주세요.');
      const kind=element('captureKind').value;
      const sequence=['rest','non_sign','transition'].includes(kind)?[]:element('captureSequence').value.split(',').map(t=>t.trim()).filter(Boolean);
      if(kind==='isolated'&&sequence.length!==1||kind==='continuous'&&!sequence.length)throw Error('수어 정답을 입력해 주세요.');
      const segments=['rest','non_sign','transition'].includes(kind)?[]:element('captureSegments').value.trim()?JSON.parse(element('captureSegments').value):[];
      const response=await fetch('/api/recordings/context');if(!response.ok)throw Error('수집 API 연결 필요 · 서버를 다시 시작해 주세요.');context=await response.json();
      const clientModel=await getModelIdentity();
      onBegin();
      const stream=getStream(),settings=stream?.getVideoTracks()[0]?.getSettings()||{};
      capture.start(stream,{session_id:element('captureSession').value,split:element('captureSplit').value,kind,sequence,segments,consent:true,
        share_slrt:kind!=='continuous'&&element('captureShareSlrt').checked,
        context,client_model:clientModel,pipeline:getPipeline(),camera:{width:settings.width||null,height:settings.height||null,frame_rate:settings.frameRate||null,mirrored:false}});
    }catch(error){showError(error);element('rawCaptureStart').disabled=false;}
  });
  element('rawCaptureStop').addEventListener('click',()=>capture.stop());
  element('rawCaptureRetry').addEventListener('click',()=>capture.retry());
  element('rawCaptureDiscard').addEventListener('click',()=>capture.discard());
  element('captureRefresh').addEventListener('click',()=>refresh().catch(showError));
  element('captureEvaluate').addEventListener('click',async()=>{
    try {
      const response=await fetch('/api/recordings/evaluation');if(!response.ok)throw Error('평가 기록을 읽지 못했습니다.');
      const report=await response.json();
      element('captureEvaluation').textContent=report.groups.length?report.groups.map(g=>`${g.split==='test'?'최종 평가':'검증'} · ${g.pipeline.startsWith('tcn')?'TCN':'DTW'} · 지문자 ${g.pipeline.endsWith('joints')?'관절 단독':'교차 확인'} · 모델 ${(g.context?.tcn_model_sha256||'미확인').slice(0,8)} · ${g.clips}개 · 단어 오류 ${g.wer===null?'미측정':(g.wer*100).toFixed(1)+'%'} · 누락 ${g.miss_rate===null?'미측정':(g.miss_rate*100).toFixed(1)+'%'} · 비수어 오출력 ${g.false_outputs_per_minute===null?'미측정':g.false_outputs_per_minute.toFixed(1)+'/분'} · 지연 p50 ${g.latency_p50_ms===null?'구간 정답 필요':g.latency_p50_ms+'ms'}`).join('\n'):'학습과 분리된 평가 촬영 없음';
      if(report.excluded?.length)element('captureEvaluation').textContent+=`\n촬영 중 모델·코드 변경으로 평가 제외 ${report.excluded.length}개`;
    }catch(error){showError(error);}
  });
  refresh().catch(()=>{});
  return capture;
}
