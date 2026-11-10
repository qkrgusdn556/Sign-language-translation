let detector=null;
async function initialize() {
  try {
    const {HandLandmarker,FilesetResolver}=await import('https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.22-rc.20250304');
    const vision=await FilesetResolver.forVisionTasks('https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.22-rc.20250304/wasm');
    const options={baseOptions:{modelAssetPath:'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',delegate:'GPU'},runningMode:'IMAGE',numHands:1,
      minHandDetectionConfidence:.7,minHandPresenceConfidence:.7,minTrackingConfidence:.6};
    try{detector=await HandLandmarker.createFromOptions(vision,options);}
    catch{options.baseOptions.delegate='CPU';detector=await HandLandmarker.createFromOptions(vision,options);}
    self.postMessage({type:'ready'});
  }catch(error){self.postMessage({type:'error',message:String(error?.message||error)});}
}
self.onmessage=({data})=>{
  const {id,image}=data;
  try{self.postMessage({id,result:detector?.detect(image)||null});}
  catch{self.postMessage({id,result:null});}
  finally{image?.close();}
};
initialize();
