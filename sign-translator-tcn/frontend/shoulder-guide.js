// Display-only reference: never drawn into recognition or recording frames.
const video = document.querySelector('#camera');
if (video) {
  const host = video.parentElement;
  const layer = document.createElement('div');
  layer.id = 'shoulderGuide';
  layer.setAttribute('aria-hidden', 'true');
  Object.assign(layer.style, {position:'absolute', pointerEvents:'none', zIndex:'3', display:'none'});
  layer.innerHTML = `<svg viewBox="0 0 1000 562.5" preserveAspectRatio="none" width="100%" height="100%" style="display:block;overflow:hidden">
    <g fill="none" stroke="#80e8d6" stroke-width="2" vector-effect="non-scaling-stroke">
      <path d="M500 285 V510" stroke-dasharray="5 9" opacity=".45"/>
      <path d="M290 285 L260 375 M710 285 L740 375" stroke-dasharray="9 7" opacity=".85"/>
      <path d="M290 285 H710" stroke-dasharray="6 8" opacity=".65"/>
      <circle cx="290" cy="285" r="12"/><circle cx="710" cy="285" r="12"/>
      <path d="M270 285 H310 M290 265 V305 M690 285 H730 M710 265 V305"/>
    </g>
    <rect x="345" y="305" width="310" height="35" rx="8" fill="#071c20" fill-opacity=".78"/>
    <text x="500" y="328" text-anchor="middle" fill="#b6fff0" font-family="system-ui,sans-serif" font-size="18">양쪽 어깨를 원에 맞춰 주세요</text>
  </svg>`;
  host.append(layer);
  const control = document.createElement('label');
  Object.assign(control.style, {position:'absolute',top:'12px',left:'12px',zIndex:'4',display:'flex',alignItems:'center',gap:'7px',background:'#0a191de0',color:'#c9fff3',padding:'7px 10px',borderRadius:'6px',fontSize:'13px',cursor:'pointer'});
  const toggle = document.createElement('input');
  toggle.type = 'checkbox';
  toggle.checked = true;
  toggle.id = 'shoulderGuideToggle';
  Object.assign(toggle.style, {width:'16px',height:'16px',margin:'0',accentColor:'#80e8d6'});
  control.append(toggle, document.createTextNode('고정 어깨 가이드'));
  host.append(control);
  function update() {
    const playing = video.readyState >= 2 && !video.paused && video.videoWidth > 0;
    layer.style.display = toggle.checked && playing ? 'block' : 'none';
    const width = video.clientWidth, height = video.clientHeight;
    const scale = Math.min(width / (video.videoWidth || 1), height / (video.videoHeight || 1));
    const w = video.videoWidth * scale, h = video.videoHeight * scale;
    // Use the displayed video rectangle, excluding object-fit: contain letterboxing.
    Object.assign(layer.style, {left:`${video.offsetLeft + (width-w)/2}px`,top:`${video.offsetTop + (height-h)/2}px`,width:`${w}px`,height:`${h}px`});
  }
  toggle.addEventListener('change', update);
  for (const event of ['playing','pause','emptied','loadedmetadata','resize','ended']) video.addEventListener(event, update);
  const observer = new ResizeObserver(update);
  observer.observe(video);
  update();
}
