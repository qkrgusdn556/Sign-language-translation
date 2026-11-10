export function createAnnotator({api, refresh, rows}) {
  const $ = id => document.getElementById(id), canvas = $('boxCanvas'), ctx = canvas.getContext('2d');
  let current, image, boxes=[], selected=-1, start=null, draft=null, dirty=false, undo=[], loading=false, saving=false;
  function checkpoint(){undo.push(structuredClone(boxes));dirty=true;}
  function paint(){
    if(!image)return;
    ctx.clearRect(0,0,canvas.width,canvas.height);ctx.drawImage(image,0,0,canvas.width,canvas.height);
    [...boxes,...(draft?[draft]:[])].forEach((b,i)=>{
      const x=b.x*canvas.width,y=b.y*canvas.height,w=b.w*canvas.width,h=b.h*canvas.height;
      ctx.strokeStyle=i===selected?'#ffd46a':'#60e4ba';ctx.lineWidth=Math.max(2,canvas.width/400);ctx.strokeRect(x,y,w,h);
      const size=Math.max(16,canvas.width/45);ctx.font=`600 ${size}px sans-serif`;
      const label=`${i+1} ${b.label}`,width=ctx.measureText(label).width+12,top=Math.max(size+6,y);
      ctx.fillStyle='#101214';ctx.fillRect(x,top-size-6,width,size+6);ctx.fillStyle='#ffffff';ctx.fillText(label,x+6,top-4);
    });
  }
  function update(){
    paint();$('boxList').replaceChildren();
    boxes.forEach((b,i)=>{const button=document.createElement('button');button.textContent=`${i+1}. ${b.label}`;button.className=i===selected?'selected':'';button.onclick=()=>{selected=i;$('boxLabel').value=b.label;update();};$('boxList').append(button);});
    $('background').checked=boxes.length===0&&current?.annotated===true&&!dirty;
    $('removeBox').disabled=selected<0;$('undoBox').disabled=!undo.length;
    for(const key of ['x','y','w','h']){const input=$(`box${key.toUpperCase()}`);input.disabled=selected<0;input.value=selected<0?'':(boxes[selected][key]*100).toFixed(2);}
  }
  async function open(row){
    if(dirty&&!confirm('저장하지 않은 박스 변경을 버릴까요?'))return;
    loading=true;dirty=false;current=row;boxes=structuredClone(row.boxes||[]);selected=-1;undo=[];image=null;
    $('boxLabel').value=row.label;ctx.clearRect(0,0,canvas.width,canvas.height);
    if(!$('annotation').open)$('annotation').showModal();
    $('annotationStatus').textContent='사진 불러오는 중';
    try{const next=new Image();next.src=`/api/images/${row.id}`;await next.decode();image=next;canvas.width=next.naturalWidth;canvas.height=next.naturalHeight;update();$('annotationStatus').textContent=`${row.label} · ${row.split==='train'?'학습용':'검증용'} · ${boxes.length}개 박스`;}
    catch(e){$('annotationStatus').textContent='사진을 불러올 수 없습니다.';}finally{loading=false;}
  }
  function point(event){const r=canvas.getBoundingClientRect();return {x:Math.max(0,Math.min(1,(event.clientX-r.left)/r.width)),y:Math.max(0,Math.min(1,(event.clientY-r.top)/r.height))};}
  canvas.onpointerdown=event=>{if(loading||saving||!image||event.button!==0)return;const label=$('boxLabel').value.trim();if(!label){$('annotationStatus').textContent='박스 글자를 입력해 주세요.';return;}start=point(event);canvas.setPointerCapture(event.pointerId);draft={...start,w:0,h:0,label};};
  canvas.onpointermove=event=>{if(!start)return;const end=point(event);draft={label:draft.label,x:Math.min(start.x,end.x),y:Math.min(start.y,end.y),w:Math.abs(start.x-end.x),h:Math.abs(start.y-end.y)};paint();};
  canvas.onpointerup=()=>{if(draft&&draft.w>=.005&&draft.h>=.005){checkpoint();boxes.push(draft);selected=boxes.length-1;$('background').checked=false;}start=null;draft=null;update();};
  canvas.onpointercancel=()=>{start=null;draft=null;paint();};
  $('boxLabel').onchange=()=>{const label=$('boxLabel').value.trim();if(selected>=0&&label){checkpoint();boxes[selected].label=label;update();}};
  $('newBox').onclick=()=>{selected=-1;update();$('boxLabel').focus();};
  for(const key of ['x','y','w','h'])$(`box${key.toUpperCase()}`).onchange=event=>{
    if(selected<0)return;const value=Number(event.target.value)/100,b={...boxes[selected],[key]:value};
    if(!Number.isFinite(value)||b.x<0||b.y<0||b.w<.005||b.h<.005||b.x+b.w>1||b.y+b.h>1){$('annotationStatus').textContent='박스가 사진 범위를 벗어납니다.';update();return;}
    checkpoint();boxes[selected]=b;update();
  };
  $('removeBox').onclick=()=>{if(selected<0)return;checkpoint();boxes.splice(selected,1);selected=-1;update();};
  $('undoBox').onclick=()=>{if(!undo.length)return;boxes=undo.pop();dirty=true;selected=-1;update();};
  $('background').onchange=()=>{if($('background').checked){if(boxes.length&&!confirm('박스를 지우고 대상 없음으로 지정할까요?')){$('background').checked=false;return;}checkpoint();boxes=[];selected=-1;update();$('background').checked=true;}};
  function close(){if(saving)return;if(dirty&&!confirm('저장하지 않은 변경을 버릴까요?'))return;dirty=false;$('annotation').close();}
  $('closeAnnotation').onclick=close;$('annotation').oncancel=event=>{event.preventDefault();close();};
  async function step(direction){if(loading||saving)return;const list=rows(),i=list.findIndex(r=>r.id===current.id),next=list[i+direction];if(next)await open(next);else $('annotationStatus').textContent=direction>0?'마지막 사진입니다.':'첫 사진입니다.';}
  $('previousPhoto').onclick=()=>step(-1);$('nextPhoto').onclick=()=>step(1);
  async function save(next){if(loading||saving||!image)return;saving=true;$('saveBoxes').disabled=true;$('saveNext').disabled=true;
    try{current=await api('annotate',{id:current.id,boxes,background:$('background').checked});dirty=false;await refresh();$('annotationStatus').textContent=`저장 완료 · ${boxes.length}개 박스`;}
    catch(e){$('annotationStatus').textContent=e.message;next=false;}finally{saving=false;$('saveBoxes').disabled=false;$('saveNext').disabled=false;}
    if(next)await step(1);
  }
  $('saveBoxes').onclick=()=>save(false);$('saveNext').onclick=()=>save(true);
  window.addEventListener('beforeunload',e=>{if(dirty){e.preventDefault();e.returnValue='';}});
  return {open};
}
