// All scene traversal here happens during initialization, never per replay car
// per frame. The native leader supplies only chase camera, audio and HUD.
export function createPolyBotSwarmWorker(context, ghosts, leader, onFailure) {
  if(typeof Worker!=='function')throw Error('browser workers unsupported');
  const interop=globalThis.polyShadeSwarm;
  const shared=interop?.apiVersion===1&&interop.capabilities().ready&&globalThis.__polybotSwarmRendererMode!=='standalone';
  if(!shared&&(typeof OffscreenCanvas!=='function'||!HTMLCanvasElement.prototype.transferControlToOffscreen))
    throw Error('OffscreenCanvas worker unsupported');
  const source=document.querySelector('canvas#screen');
  if(!source||!context.scene||!leader.car?.polybotReplayRoot)throw Error('native scene/canvas unavailable');
  const generation=(globalThis.__polybotSwarmWorkerGeneration=(globalThis.__polybotSwarmWorkerGeneration||0)+1);
  const canonical={data:null,offsets:new Uint32Array(ghosts.length+1)};
  let samples=0;
  for(let c=0;c<ghosts.length;c++){canonical.offsets[c]=samples;samples+=ghosts[c].samples.length;}
  canonical.offsets[ghosts.length]=samples;canonical.data=new Float64Array(samples*53);
  for(let c=0;c<ghosts.length;c++)for(let i=0;i<ghosts[c].samples.length;i++){
    const sample=ghosts[c].samples[i],offset=(canonical.offsets[c]+i)*53;
    if(!sample.wheelState)throw Error('legacy wheel reconstruction requires native renderer');
    for(let j=0;j<9;j++)canonical.data[offset+j]=sample[j];
    for(let j=0;j<42;j++)canonical.data[offset+9+j]=sample.wheelState[j];
    canonical.data[offset+51]=Number(sample.hudFrame?.controls?.applied?.throttle)||0;
    canonical.data[offset+52]=Number(sample.hudFrame?.controls?.applied?.brake)||0;
  }
  function colourData(){const result=new Float32Array(ghosts.length*3);
    for(let c=0;c<ghosts.length;c++){const hex=Number.parseInt(ghosts[c].color.slice(1),16);
      // PolyTrack's car-style uniforms store byte channels directly; applying
      // an extra sRGB conversion changes all intermediate generation colours.
      for(let j=0;j<3;j++)result[c*3+j]=((hex>>(16-j*8))&255)/255;}
    return result;}
  const root=leader.car.polybotReplayRoot,originalParent=root.parent;
  const saved=leader.car.getCarState(),base=structuredClone(saved);
  base.position={x:0,y:0,z:0};base.quaternion={x:0,y:0,z:0,w:1};base.steering=0;
  base.wheelSuspensionLength=[0,0,0,0];base.wheelDeltaRotation=[0,0,0,0];base.wheelContact=[null,null,null,null];
  leader.car.setCarState(base,true);leader.car.update(0);root.updateMatrixWorld(true);
  const T=context.three, groupsByKey=new Map(),mounts=new Float64Array(12);
  let wheelIndex=0;
  function normal(array,index,matrix){let x=array.getX(index),y=array.getY(index),z=array.getZ(index);
    const a=matrix.elements,px=a[0]*x+a[4]*y+a[8]*z,py=a[1]*x+a[5]*y+a[9]*z,pz=a[2]*x+a[6]*y+a[10]*z;
    const inv=1/(Math.hypot(px,py,pz)||1);return [px*inv,py*inv,pz*inv];}
  root.traverse(object=>{
    if(!object.isMesh||!object.visible)return;
    for(let p=object.parent;p&&p!==root;p=p.parent)if(!p.visible)return;
    const kind=object.name.startsWith('Wheel')?1:object.name==='Suspension'?2:0;
    if(kind===1){if(wheelIndex===0)mounts.set(object.position.toArray(),0);
      else mounts.set(object.position.toArray(),wheelIndex*3);wheelIndex++;if(wheelIndex>1)return;}
    const geometry=object.geometry,position=geometry.attributes.position,normals=geometry.attributes.normal;
    if(!position||!normals)throw Error('car geometry lacks positions/normals');
    const materials=Array.isArray(object.material)?object.material:[object.material];
    const parts=geometry.groups.length?geometry.groups:[{start:0,count:geometry.index?.count??position.count,materialIndex:0}];
    for(const part of parts){const material=materials[part.materialIndex||0];if(!material||material.opacity===0)continue;
      if(material.transparent&&material.opacity!==1)throw Error('unsupported native transparent car material');
      const key=`${kind}:${material.name}`;let group=groupsByKey.get(key);
      if(!group){let rgb=material.color?.toArray?.()||[.1,.1,.1];

        group={kind,name:material.name,colour:new Float32Array(rgb),paintMode:material.name==='Main'?1:material.name==='BrakeLight'?2:material.name==='Tire'?3:0,
          materialType:material.type,flatShading:material.flatShading,roughness:material.roughness,metalness:material.metalness,
          envMapIntensity:material.envMapIntensity,clearcoat:material.clearcoat,clearcoatRoughness:material.clearcoatRoughness,
          emissive:material.emissive?.toArray?.(),emissiveIntensity:material.emissiveIntensity,
          specularIntensity:material.specularIntensity,specularColor:material.specularColor?.toArray?.(),ior:material.ior,
          side:material.side,shadowSide:material.shadowSide,toneMapped:material.toneMapped,fog:material.fog,
          shininess:material.shininess,specular:material.specular?.toArray?.(),reflectivity:material.reflectivity,
          positions:[],normals:[],morphs:kind===2?[[],[],[],[]]:null,
          normalMorphs:kind===2&&geometry.morphAttributes.normal?.length===4?[[],[],[],[]]:null};groupsByKey.set(key,group);}
      for(let j=part.start;j<part.start+part.count;j++){
        const index=geometry.index?geometry.index.getX(j):j;
        let x=position.getX(index),y=position.getY(index),z=position.getZ(index);
        if(kind!==1){const a=object.matrixWorld.elements;const px=a[0]*x+a[4]*y+a[8]*z+a[12];
          const py=a[1]*x+a[5]*y+a[9]*z+a[13];z=a[2]*x+a[6]*y+a[10]*z+a[14];x=px;y=py;}
        group.positions.push(x,y,z);
        if(kind===1)group.normals.push(normals.getX(index),normals.getY(index),normals.getZ(index));
        else group.normals.push(...normal(normals,index,object.matrixWorld));
        if(kind===2)for(let k=0;k<4;k++){
          const target=geometry.morphAttributes.position[k];
          if(!target||!geometry.morphTargetsRelative)throw Error('unsupported suspension morph geometry');
          group.morphs[k].push(target.getX(index),target.getY(index),target.getZ(index));
          if(group.normalMorphs){const n=geometry.morphAttributes.normal[k];group.normalMorphs[k].push(n.getX(index),n.getY(index),n.getZ(index));}
        }
      }
    }
  });
  if(wheelIndex!==4)throw Error('native wheel geometry contract changed');
  leader.car.setCarState(saved,true);leader.car.update(0);
  const groups=[...groupsByKey.values()].map(g=>({...g,positions:new Float32Array(g.positions),normals:new Float32Array(g.normals),morphs:g.morphs?.map(a=>new Float32Array(a)),normalMorphs:g.normalMorphs?.map(a=>new Float32Array(a))}));
  if(groups.length>24)throw Error('car uses too many material groups for packed worker protocol');
  let geometryBytes=0;for(const g of groups)geometryBytes+=g.positions.byteLength+g.normals.byteLength+(g.morphs?.reduce((n,a)=>n+a.byteLength,0)||0);
  const scene=context.scene,occluderObjects=[],sunLights=[],ambientLights=[];
  scene.updateMatrixWorld(true);
  const isCar=o=>{for(let p=o;p;p=p.parent)if(p.userData?.polybotCar)return true;return false;};
  let depthVertices=0;
  scene.traverse(o=>{
    if(o.isDirectionalLight&&!o.userData?.polyShadeOwned)sunLights.push(o);
    if(o.isAmbientLight||o.isHemisphereLight)ambientLights.push(o);
    if(shared||!o.isMesh||isCar(o)||o.userData?.polyShadeOwned||o.name==='PolyShade sky')return;
    for(let p=o;p;p=p.parent)if(!p.visible)return;
    if(!o.geometry?.attributes.position)return;
    // Alpha-tested/moving world meshes need a separate adapter; never silently
    // turn fences, foliage or animated surfaces into opaque occluders.
    const materials=Array.isArray(o.material)?o.material:[o.material];
    const parts=o.geometry.groups.length?o.geometry.groups:[{start:0,count:o.geometry.index?.count??o.geometry.attributes.position.count,materialIndex:0}];
    const opaque=parts.filter(part=>{const m=materials[part.materialIndex||0];return m&&!m.transparent&&m.opacity!==0&&!m.alphaTest;});
    if(!opaque.length)return;
    occluderObjects.push({object:o,parts:opaque});
    depthVertices+=opaque.reduce((n,p)=>n+p.count,0)*(o.isInstancedMesh?o.count:1);
  });
  const occluders=new Float32Array(depthVertices*3),matrix=new T.kn4(),instance=new T.kn4();let cursor=0;
  for(const {object,parts}of occluderObjects){const p=object.geometry.attributes.position,index=object.geometry.index;
    for(let i=0;i<(object.isInstancedMesh?object.count:1);i++){
      if(object.isInstancedMesh){object.getMatrixAt(i,instance);matrix.multiplyMatrices(object.matrixWorld,instance);}else matrix.copy(object.matrixWorld);
      const a=matrix.elements;
      for(const part of parts)for(let j=part.start;j<part.start+part.count;j++){
        const k=index?index.getX(j):j,x=p.getX(k),y=p.getY(k),z=p.getZ(k);
        occluders[cursor++]=a[0]*x+a[4]*y+a[8]*z+a[12];occluders[cursor++]=a[1]*x+a[5]*y+a[9]*z+a[13];occluders[cursor++]=a[2]*x+a[6]*y+a[10]*z+a[14];
      }
    }
  }
  geometryBytes+=occluders.byteLength;
  const canvas=shared?null:document.createElement('canvas');
  if(canvas){canvas.id='polybot-worker-swarm';canvas.setAttribute('aria-label','PolyBot worker swarm');
  Object.assign(canvas.style,{position:'fixed',pointerEvents:'none',background:'transparent',zIndex:'0'});
  (document.querySelector('#polyshade-webgpu')||source).after(canvas);
  }
  function resize(){if(!canvas)return;const rect=source.getBoundingClientRect();Object.assign(canvas.style,{left:`${rect.left}px`,top:`${rect.top}px`,width:`${rect.width}px`,height:`${rect.height}px`});}
  const observer=new ResizeObserver(resize);observer.observe(source);window.addEventListener('resize',resize);document.addEventListener('fullscreenchange',resize);resize();
  const url=URL.createObjectURL(new Blob([`${createSwarmKernel.toString()}\n${createSwarmTrails.toString()}\n(${polybotSwarmWorker.toString()})();`],{type:'text/javascript'}));
  let worker,alive=true,ready=false,failure=null,stats={backend:'initializing worker'},mainMs=0,mainFrames=0;
  const pool=[new Float64Array(64),new Float64Array(64),new Float64Array(64)];
  let settings={},lastSettings=-Infinity,epoch=0,lastClock=-1,sequence=0,handle=null,outputOffsets=null,trailOffset=0,renderClock=0,renderEpoch=-1,pendingFrames=[];
  const fallbackRestore=()=>{
    for(let c=0;c<ghosts.length;c++)if(!ghosts[c].samples){const data=[];let distance=0;
      for(let r=canonical.offsets[c];r<canonical.offsets[c+1];r++){const a=r*53,row=Array.from(canonical.data.subarray(a,a+9));
        row.wheelState=Array.from(canonical.data.subarray(a+9,a+51));
        if(data.length){const p=data[data.length-1];distance+=Math.hypot(row[2]-p[2],row[3]-p[3],row[4]-p[4]);}
        row.distanceM=distance;data.push(row);}
      ghosts[c].samples=data;
    }
  };
  function dispose(){if(!alive)return;alive=false;clearTimeout(timeout);observer.disconnect();window.removeEventListener('resize',resize);document.removeEventListener('fullscreenchange',resize);
    canvas?.remove();if(handle!==null)interop.unregisterSwarmSource(handle);if(worker){worker.postMessage({type:'dispose',generation});const old=worker;setTimeout(()=>old.terminate(),100);}
    if(originalParent&&!root.parent)originalParent.add(root);URL.revokeObjectURL(url);}
  function fail(reason){if(!alive)return;failure=reason;fallbackRestore();dispose();onFailure(reason);}
  let timeout;
  try{
    worker=new Worker(url,{name:'PolyBot renderer-only swarm'});
    worker.onmessage=event=>{const m=event.data;if(!alive||m.generation!==generation)return;
      if(m.type==='ready'){ready=true;clearTimeout(timeout);stats=m;outputOffsets=m.outputOffsets;trailOffset=m.trailOffset;
        // Retain one packed canonical copy for crash fallback. Drop the much
        // larger nested JS sample graphs; worker owns its transferred copy.
        for(const ghost of ghosts)if(ghost!==leader)ghost.samples=null;
      }else if(m.type==='instances'){
        // Stage until the replay render boundary. Updating GPU transforms from
        // an asynchronous message can leave the chase camera one frame behind.
        if(pendingFrames.length>=3){fail('swarm output pool exceeded');return;}
        pendingFrames.push(m);
      }else if(m.type==='recycle')pool.push(new Float64Array(m.buffer));
      else if(m.type==='stats'){stats=m.stats;for(let c=0;c<ghosts.length;c++)if(ghosts[c]!==leader){
        const g=ghosts[c],o=c*4;g.position.x=m.positions[o];g.position.y=m.positions[o+1];g.position.z=m.positions[o+2];g.visible=!!m.positions[o+3];}}
      else if(m.type==='error')fail(m.error);
    };
    worker.onerror=event=>{event.preventDefault();fail(event.message||'swarm worker crashed');};
    const offscreen=canvas?.transferControlToOffscreen(),copy=canonical.data.slice(),offsets=canonical.offsets.slice(),colours=colourData();
    const transfers=[copy.buffer,offsets.buffer,colours.buffer,mounts.buffer];
    if(shared)handle=interop.registerSwarmSource({version:1,maxCars:ghosts.length,groups,scene:context.scene,onUnavailable:fail});
    else {
      transfers.push(offscreen,occluders.buffer);
      for(const g of groups){transfers.push(g.positions.buffer,g.normals.buffer,g.colour.buffer);if(g.morphs)for(const a of g.morphs)transfers.push(a.buffer);
        if(g.normalMorphs)for(const a of g.normalMorphs)transfers.push(a.buffer);}
    }
    originalParent?.remove(root);
    worker.postMessage({type:'init',generation,canvas:offscreen,shared,replay:{data:copy,offsets},colours,groups:shared?groups.map(g=>({kind:g.kind})):groups,occluders:shared?undefined:occluders,mounts,geometryBytes},transfers);
    timeout=setTimeout(()=>fail('swarm worker initialization timed out'),15000);
  }catch(error){dispose();throw error;}
  function frame(time,playing,opacity,endBehavior,fadeDuration,reset=false){
    if(!alive||!ready||!pool.length)return;const begin=performance.now(),p=pool.pop(),camera=context.getCamera();
    camera.updateMatrixWorld(true);
    if(reset||time<lastClock)epoch++;lastClock=time;
    p[0]=time;p[1]=playing?1:0;p[2]=epoch;p[3]=opacity;p[4]=endBehavior==='freeze'?1:endBehavior==='fade'?2:0;
    p[5]=fadeDuration;p[6]=source.width;p[7]=source.height;p[57]=++sequence;
    for(let j=0;j<16;j++){p[8+j]=camera.matrixWorldInverse.elements[j];p[24+j]=camera.projectionMatrix.elements[j];}
    if(!shared){
    // Cache settings only at 2 Hz; no per-car material conversion or scene scan.
    if(begin-lastSettings>500){lastSettings=begin;try{const saved=JSON.parse(localStorage.getItem('polyshade.settings')||'{}');settings=saved.enabled?saved.overrides||{}:{};}catch{settings={};}}
    const sun=sunLights.find(l=>l.visible)||sunLights[0];
    let dx=sun?.position.x??-1,dy=sun?.position.y??2,dz=sun?.position.z??1;
    if(sun?.target){dx-=sun.target.position.x;dy-=sun.target.position.y;dz-=sun.target.position.z;}
    const length=Math.hypot(dx,dy,dz)||1;p[40]=dx/length;p[41]=dy/length;p[42]=dz/length;
    p[43]=(sun?.color.r??1)*1.05;p[44]=(sun?.color.g??.85)*1.05;p[45]=(sun?.color.b??.65)*1.05;
    p[46]=.42;p[47]=.48;p[48]=.58;
    for(const a of ambientLights)if(a.visible){p[46]+=a.color.r*a.intensity*.2;p[47]+=a.color.g*a.intensity*.2;p[48]+=a.color.b*a.intensity*.2;}
    p[49]=camera.matrixWorld.elements[12];p[50]=camera.matrixWorld.elements[13];p[51]=camera.matrixWorld.elements[14];
    p[52]=.47;p[53]=.60;p[54]=.67;p[55]=settings.exposure??1.06;p[56]=settings.atmosphereStrength??.13;
    }
    worker.postMessage({type:'frame',generation,buffer:p.buffer},[p.buffer]);mainMs+=performance.now()-begin;mainFrames++;
  }
  function present(){
    if(!pendingFrames.length||!alive)return;
    try {
      for(const m of pendingFrames){
      if(m.epoch===epoch){const values=new Float32Array(m.buffer);
        interop.updateSwarmInstances(handle,values,outputOffsets,values.subarray(0,outputOffsets.length));
        if(interop.updateSwarmTrails)interop.updateSwarmTrails(handle,values,trailOffset,values[28],values[29]===1);
        renderClock=m.time;renderEpoch=m.epoch;
      }
      worker.postMessage({type:'instances_recycle',generation,buffer:m.buffer},[m.buffer]);
      }
      pendingFrames.length=0;
    }catch(error){fail(String(error.message||error));}
  }
  return {frame,present,dispose,renderClock:()=>shared?renderClock:null,isFresh:()=>!shared||renderEpoch===epoch,invalidate(){epoch++;},restore:fallbackRestore,updateColours(){const colours=colourData();worker.postMessage({type:'colours',generation,colours},[colours.buffer]);},
    diagnostics(){return {...stats,ready,failure,mainThreadMs:mainFrames?mainMs/mainFrames:0,
      canonicalBytes:canonical.data.byteLength,geometryBytes,inFlight:3-pool.length,sharedMemory:false,
      interop:shared?interop.diagnostics():null,renderClock};}};
}
