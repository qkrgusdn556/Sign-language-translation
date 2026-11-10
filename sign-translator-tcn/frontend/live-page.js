// Share the laboratory controls and recognition engine without duplicating IDs or logic.
try {
  const response=await fetch('/frontend/index.html',{cache:'no-store'});
  if(!response.ok)throw Error('인식 화면을 불러오지 못했습니다.');
  const template=new DOMParser().parseFromString(await response.text(),'text/html');
  const main=template.querySelector('main');
  document.body.append(main,template.querySelector('#clearDialog'));
  main.querySelector('h1').textContent='실시간 수어';
  const link=main.querySelector('.live-link');link.href='/';link.textContent='학습·관리';
  const hidden=document.createElement('div');hidden.hidden=true;main.append(hidden);
  const dialog=document.createElement('dialog');dialog.id='liveSettings';
  dialog.innerHTML='<div class="section-head"><h2>인식 연결 설정</h2><button id="closeLiveSettings" type="button">닫기</button></div>';
  document.body.append(dialog);
  const stage=main.querySelector('.stage'),sentence=main.querySelector('.sentence-panel');
  const cameraTools=document.createElement('div');cameraTools.className='live-camera-tools';
  cameraTools.append(document.querySelector('#startBtn'));
  const settings=document.createElement('button');settings.type='button';settings.textContent='모델·인식 설정';
  settings.onclick=()=>dialog.showModal();cameraTools.append(settings);
  const tracking=document.createElement('span');tracking.className='live-tracking';tracking.append(document.querySelector('#trackingState'));
  cameraTools.append(tracking);stage.append(cameraTools);
  document.querySelector('#closeLiveSettings').onclick=()=>dialog.close();
  dialog.append(main.querySelector('.yolo-panel'));
  const diagnostics=document.createElement('div');diagnostics.className='live-jamo-status';
  const diagnosticsLabel=document.createElement('strong');diagnosticsLabel.textContent='지문자 인식 상태';
  diagnostics.append(diagnosticsLabel,document.querySelector('#yoloResult'));
  stage.append(diagnostics);
  dialog.append(sentence.querySelector('.section-head'));
  hidden.append(document.querySelector('#trainTcnBtn'));
  const fingerspell=sentence.querySelector('.fingerspell-controls');
  for(const label of [...fingerspell.querySelectorAll(':scope > label')])dialog.append(label);
  dialog.append(sentence.querySelector('details'));
  const liveColumns=document.createElement('div');liveColumns.className='live-columns';
  main.querySelector('.app-header').after(liveColumns);liveColumns.append(stage,sentence);
  hidden.append(stage.querySelector('.result-panel'));
  for(const section of [...main.querySelectorAll(':scope > section')])hidden.append(section);
  document.querySelector('#liveLoading').remove();
  await import('./app.js?v=yolo-only-1');
  await import('./shoulder-guide.js?v=2');
  await import('./sentence_polish.js?v=gemini-auto-700-1');
  await import('./sentence_speech.js?v=1');
}catch(error){
  const message=document.querySelector('#liveLoading')||document.body.appendChild(document.createElement('p'));
  message.textContent=`화면 준비 실패: ${error.message} 새로고침해 주세요.`;
}
