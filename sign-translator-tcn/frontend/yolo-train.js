import { createAnnotator } from './yolo-annotate.js?v=boxes-1';
const $ = id => document.getElementById(id);
let stream = null, snapshot = null, galleryKey = '', historyKey = '', busy = false;
const selectedImages = new Set();
let removingImages = false;
function updateSelection() {
  $('selectionCount').textContent = `${selectedImages.size}장 선택`;
  $('deleteSelected').disabled = removingImages || selectedImages.size === 0;
  $('selectVisible').disabled = removingImages || !snapshot?.images.length;
  $('clearSelection').disabled = removingImages || selectedImages.size === 0;
}
const notice = message => { $('notice').textContent = message; };
const annotator = createAnnotator({api, refresh, rows:()=>snapshot.images.filter(row=>!$('filter').value||row.label===$('filter').value)});
async function api(path, value) {
  const response = await fetch(`/api/${path}`, value === undefined ? {} : {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(value)});
  const body = await response.json();
  if (!response.ok) throw new Error(body.error || '요청에 실패했습니다.');
  return body;
}
function download(run, name, label) {
  const a = document.createElement('a'); a.href = `/api/runs/${run}/${name}`; a.download = name; a.textContent = label; return a;
}
function renderGallery() {
  const gallery = $('gallery'); gallery.replaceChildren();
  const rows = snapshot.images.filter(row => !$('filter').value || row.label === $('filter').value);
  if (!rows.length) { const p = document.createElement('p'); p.className='empty'; p.textContent='저장된 사진이 없습니다.'; gallery.append(p); }
  for (const row of rows.slice().reverse()) {
    const card = document.createElement('article'); card.className='photo';
    const selection=document.createElement('label'); selection.className='photo-select';
    const checkbox=document.createElement('input'); checkbox.type='checkbox'; checkbox.checked=selectedImages.has(row.id); checkbox.disabled=removingImages;
    checkbox.setAttribute('aria-label',`${row.label} 사진 선택 ${row.id}`);
    checkbox.onchange=()=>{if(checkbox.checked)selectedImages.add(row.id);else selectedImages.delete(row.id);updateSelection();};
    selection.append(checkbox,document.createTextNode('선택'));
    const img = document.createElement('img'); img.src=`/api/images/${row.id}`; img.alt=row.label; img.loading='lazy';
    const footer = document.createElement('footer'), text = document.createElement('span'), remove = document.createElement('button');
    text.textContent=`${row.label} · ${row.split === 'train' ? '학습' : '검증'} · ${row.annotated ? `박스 ${row.boxes.length}` : '미작업'}`;
    const edit=document.createElement('button');edit.textContent='박스 편집';edit.onclick=()=>annotator.open(row);
    remove.textContent='×'; remove.title='사진 삭제'; remove.setAttribute('aria-label', `${row.label} 사진 삭제`);
    remove.onclick=async()=>{if(!confirm('이 사진을 삭제할까요? 완료된 학습 모델은 유지됩니다.'))return;try{await api('remove',{id:row.id});await refresh();}catch(e){notice(e.message);}};
    footer.append(text,edit,remove); card.append(selection,img,footer); gallery.append(card);
  }
  updateSelection();
}
async function refresh() {
  snapshot = await api('status');
  const existingIds=new Set(snapshot.images.map(row=>row.id));
  for(const id of selectedImages)if(!existingIds.has(id))selectedImages.delete(id);
  updateSelection();
  const labels = [...new Set(snapshot.images.map(row=>row.label))].sort();
  const key = JSON.stringify(snapshot.images);
  if (key !== galleryKey) {
    galleryKey=key; const selected=$('filter').value;
    $('filter').replaceChildren(new Option('전체',''),...labels.map(label=>new Option(label,label)));
    $('filter').value=labels.includes(selected)?selected:'';
    $('labels').replaceChildren(...labels.map(label=>new Option(label,label)));
    $('summary').replaceChildren(...labels.map(label=>{const span=document.createElement('span');span.textContent=`${label} · 학습 ${snapshot.images.filter(r=>r.label===label&&r.split==='train').length} / 검증 ${snapshot.images.filter(r=>r.label===label&&r.split==='val').length}`;return span;}));
    renderGallery();
  }
  $('counts').textContent=`${snapshot.images.length}장 · ${labels.length}종류`;
  $('annotatedCount').textContent=`${snapshot.images.filter(row=>row.annotated).length} / ${snapshot.images.length}장`;
  $('modelRequirement').hidden=$('task').value!=='detect'||snapshot.detectionModel===true;
  const job=snapshot.job, running=['running','cancelling'].includes(job.state);
  $('train').disabled=running || busy; $('cancel').disabled=!running || job.state==='cancelling';
  const names={idle:'학습 대기',running:'학습 중',complete:'학습 완료',error:'학습 실패',cancelled:'학습 중단됨',cancelling:'중단 중'};
  $('jobState').textContent=job.progress?.phase==='exporting'&&running?'ONNX 저장 중':`${names[job.state]}${running&&job.progress?.epoch ? ` · ${job.progress.epoch}/${job.epochs}`:''}`;
  $('progress').value=job.state==='complete'?100:(job.progress?.epoch||0)/(job.epochs||1)*100;
  $('error').hidden=!job.error; $('error').textContent=job.error||'';
  $('downloads').replaceChildren();
  if(job.run) $('downloads').append(download(job.run,'training.log','학습 로그 저장'));
  const history=JSON.stringify(snapshot.runs);
  if(history!==historyKey){historyKey=history;$('history').replaceChildren();
    if(!snapshot.runs.length) $('history').textContent='아직 완료된 학습이 없습니다.';
    for(const run of snapshot.runs){const section=document.createElement('div');section.className='run';const title=document.createElement('strong');title.textContent=(run.model || 'YOLO11')+' · '+run.labels.join(' · ');const result=document.createElement('p');result.textContent=run.task==='detect'?`손 박스 탐지 · 검증 ${run.total}장 · mAP50 ${(run.map50*100).toFixed(1)}% / mAP50–95 ${(run.map50_95*100).toFixed(1)}%`:`사진 분류 · 검증 ${run.total}장 중 ${run.correct}장 일치 (${Math.round(run.correct/run.total*100)}%)`;
      const warning=document.createElement('p');warning.className='warning';warning.textContent='이 검증 자료에서의 결과이며, 실제 수어 인식 정확도를 보장하지 않습니다.';
      section.append(title,result,warning,download(run.id,'fingerspell.onnx','ONNX 모델 저장'),download(run.id,'labels.json','라벨 저장'));$('history').append(section);}
  }
}
function stop(){stream?.getTracks().forEach(track=>track.stop());stream=null;$('camera').srcObject=null;$('start').disabled=false;$('stop').disabled=true;$('capture').disabled=true;$('cameraState').textContent='카메라 꺼짐';}
$('start').onclick=async()=>{try{$('start').disabled=true;stream=await navigator.mediaDevices.getUserMedia({video:{width:{ideal:1280},height:{ideal:720}},audio:false});$('camera').srcObject=stream;await $('camera').play();$('stop').disabled=false;$('capture').disabled=false;$('cameraState').textContent='촬영 준비';notice('');}catch(e){stop();notice(`카메라를 켤 수 없습니다: ${e.message}`);}};
$('stop').onclick=stop;
window.addEventListener('pagehide',stop);
function asBase64(blob){return new Promise((resolve,reject)=>{const reader=new FileReader();reader.onload=()=>resolve(reader.result.split(',')[1]);reader.onerror=reject;reader.readAsDataURL(blob);});}
async function save(blob,label,split){if(blob.size>8*1024*1024)throw new Error('사진은 8MB 이하여야 합니다.');await api('images',{label,split,image:await asBase64(blob)});}
async function collect(action){if(busy)return;const label=$('label').value.trim(),split=$('split').value;if(!label){notice('수어 / 지문자 이름을 입력해 주세요.');$('label').focus();return;}busy=true;try{await action(label,split);notice('사진을 저장했습니다.');await refresh();}catch(e){notice(e.message);}finally{busy=false;}}
$('capture').onclick=()=>collect(async(label,split)=>{const video=$('camera');if(!video.videoWidth)throw new Error('카메라 화면을 기다려 주세요.');const canvas=document.createElement('canvas');canvas.width=video.videoWidth;canvas.height=video.videoHeight;canvas.getContext('2d').drawImage(video,0,0);const blob=await new Promise(resolve=>canvas.toBlob(resolve,'image/jpeg',.95));await save(blob,label,split);});
$('upload').onchange=()=>{const files=Array.from($('upload').files);collect(async(label,split)=>{let saved=0;try{for(const file of files){await save(file,label,split);saved++;notice(`${saved}/${files.length}장 저장 중`);}}catch(e){await refresh();throw new Error(`${saved}장 저장됨 · ${e.message}`);}finally{$('upload').value='';}});};
$('filter').onchange=()=>{selectedImages.clear();renderGallery();};
$('selectVisible').onclick=()=>{
  for(const row of snapshot.images)if(!$('filter').value||row.label===$('filter').value)selectedImages.add(row.id);
  renderGallery();
};
$('clearSelection').onclick=()=>{selectedImages.clear();renderGallery();};
$('deleteSelected').onclick=async()=>{
  if(removingImages||!selectedImages.size)return;
  const ids=[...selectedImages];
  if(!confirm(`선택한 사진 ${ids.length}장과 라벨링을 목록에서 삭제할까요? 완료된 학습 모델은 유지됩니다.`))return;
  removingImages=true;renderGallery();
  try{
    const result=await api('remove-many',{ids});
    selectedImages.clear();
    notice(`${result.removed}장 삭제 완료 · 완료된 학습 모델은 유지됩니다.`);
  }catch(error){notice(error.message);}
  finally{removingImages=false;await refresh().catch(error=>notice(error.message));renderGallery();}
};
$('task').onchange=()=>{$('inputSize').textContent=$('task').value==='detect'?'손 박스 · 320 × 320':'전체 화면 · 224 × 224';};
$('train').onclick=async()=>{if(busy)return;busy=true;$('train').disabled=true;try{await api('train',{epochs:Number($('epochs').value),task:$('task').value});notice('학습을 시작했습니다. 기존 수어 인식 모델은 바뀌지 않습니다.');}catch(e){notice(e.message);}finally{busy=false;await refresh().catch(e=>notice(e.message));}};
$('cancel').onclick=async()=>{try{await api('cancel',{});await refresh();}catch(e){notice(e.message);}};
async function poll(){try{await refresh();}catch(e){notice(`학습 서버 연결 실패: ${e.message}`);}setTimeout(poll,2000);}
notice('학습용과 검증용 사진은 서로 다른 촬영에서 수집해 주세요.');poll();
