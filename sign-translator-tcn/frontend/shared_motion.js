import {createRawCapture} from './raw_capture.js?v=shared-motion-1';

export function installSharedMotion({getStream,getVideo,getPipeline,getModelIdentity,onChange=()=>{},onCommitted=async()=>{},beforeTraining=()=>{}}) {
  const el=id=>document.getElementById(id);
  let preparing=false,refreshing=false,training=false,canTrain=false,epoch=0,motion=null,lastResponse=null,timer=null;
  async function request(url,body) {
    const response=await fetch(url,body?{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}:{});
    const value=await response.json();if(!response.ok)throw Error(value.error||'공동 저장 연결 실패');return value;
  }
  const capture=createRawCapture({post:async body=>{lastResponse=await request('/api/shared-motion',body);return lastResponse;},
    onState({phase,message}) {
      el('sharedMotionStatus').textContent=message;
      el('sharedMotionRetry').hidden=phase!=='error';el('sharedMotionDiscard').hidden=phase!=='error';update();
    },
    onSaved() {
      refreshing=true;update();
      Promise.resolve(onCommitted(lastResponse)).then(()=>{
        el('sharedMotionStatus').textContent=lastResponse?.slrt?.status==='error'
          ?`관절·영상 저장됨 · SLRT 전달 실패: ${lastResponse.slrt.error}`:'관절·영상 공동 저장 완료 · 모델 학습 대기';
      }).catch(error=>{el('sharedMotionStatus').textContent=`영상 저장됨 · 샘플 목록 새로고침 필요: ${error.message}`;})
        .finally(()=>{refreshing=false;update();refresh();});
    },
  });
  function update() {
    const busy=preparing||refreshing||training||capture.phase()!=='idle';
    el('shareMotionVideo').disabled=busy;
    el('slrtTrainBtn').disabled=busy||!canTrain;
    el('slrtSyncBtn').disabled=busy;
    onChange();
  }
  async function refresh() {
    clearTimeout(timer);
    try {
      const state=await request('/api/slrt/status');training=['starting','running'].includes(state.job?.status);canTrain=state.can_train;
      const message=!training&&state.pending_training_clips>0?'저장한 영상의 학습 시작 필요':state.job?.message||'학습 대기';
      el('slrtSharedStatus').textContent=`공동 저장 ${state.shared_clips}개 · 학습 미반영 ${state.pending_training_clips}개 · ${message}`;
      el('slrtTrainBtn').title=state.can_train?'저장된 학습용 영상으로 영상·관절 모델을 학습합니다.':'각 수어와 배경 동작의 학습 영상이 각각 두 개 이상 필요합니다.';
      update();
      if(training)timer=setTimeout(refresh,2000);
    }catch(error){canTrain=false;el('slrtSharedStatus').textContent=`SLRT 연결 대기: ${error.message}`;update();}
  }
  el('sharedMotionRetry').addEventListener('click',()=>capture.retry());
  el('shareMotionVideo').addEventListener('change',update);
  el('sharedMotionDiscard').addEventListener('click',()=>{capture.discard();update();});
  el('slrtRefreshBtn').addEventListener('click',refresh);
  el('slrtSyncBtn').addEventListener('click',async()=>{
    if(api.busy())return;
    preparing=true;update();
    try{const result=await request('/api/slrt/sync',{sync:true});const failures=result.results.filter(row=>row.status==='error');
      el('sharedMotionStatus').textContent=failures.length?`SLRT 전달 실패 ${failures.length}개 · ${failures[0].error}`:'SLRT 자료 전달 완료 · 모델 학습 대기';
    }catch(error){el('sharedMotionStatus').textContent=error.message;}
    finally{preparing=false;update();refresh();}
  });
  el('slrtTrainBtn').addEventListener('click',async()=>{
    if(api.busy())return;
    preparing=true;update();
    try{beforeTraining();await request('/api/slrt/train',{train:true});training=true;el('slrtSharedStatus').textContent='SLRT 영상 모델 학습 시작 요청됨';timer=setTimeout(refresh,1000);}
    catch(error){el('slrtSharedStatus').textContent=`학습 시작 실패: ${error.message}`;}
    finally{preparing=false;update();}
  });
  const api={
    busy:()=>preparing||refreshing||training||capture.phase()!=='idle',
    canTrain:()=>canTrain&&!api.busy(),
    active:()=>capture.active(),
    async begin(label,{rest=false}={}) {
      if(api.busy())throw Error('공동 저장·학습 또는 미저장 영상을 먼저 처리해 주세요.');
      if(el('captureSplit').value!=='train')throw Error('공동 동작 수집은 자료 구분을 학습으로 선택해 주세요. 평가는 원본 촬영을 사용하세요.');
      if(label.includes(','))throw Error('동작 하나의 수어 이름에는 쉼표를 넣지 마세요.');
      const stream=getStream(),token=++epoch;
      preparing=true;el('sharedMotionStatus').textContent='공동 촬영 준비 중';update();
      try {
        const context=await request('/api/recordings/context'),identity=await getModelIdentity();
        if(epoch!==token||stream!==getStream())throw Error('공동 촬영 시작이 취소되었습니다.');
        const settings=stream?.getVideoTracks()[0]?.getSettings()||{},video=getVideo();
        motion={label,sequence:[]};
        capture.start(stream,{session_id:el('captureSession').value,split:'train',kind:rest?'rest':'isolated',sequence:rest?[]:[label],segments:[],
          consent:true,share_slrt:true,motion_sample:motion,context,client_model:identity,pipeline:getPipeline(),
          camera:{width:settings.width||video.videoWidth,height:settings.height||video.videoHeight,frame_rate:settings.frameRate||null,mirrored:false}});
      }finally{preparing=false;update();}
    },
    observe:frame=>capture.observe(frame),
    finish(frames) {motion.sequence=frames.map(frame=>[...frame]);capture.stop();},
    cancel() {epoch++;capture.cancel();},
    refresh,
  };
  refresh();return api;
}
