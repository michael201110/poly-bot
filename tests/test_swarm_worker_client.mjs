import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import vm from 'node:vm';

function fixture() {
  const identity=[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1];
  const attribute={count:3,getX:i=>i===1?1:0,getY:i=>i===2?1:0,getZ:()=>0};
  const material={name:'Main',type:'MeshStandardMaterial',flatShading:true,
    roughness:.4,metalness:0,envMapIntensity:1,color:{toArray:()=>[1,1,1]},opacity:1};
  const parent={remove(root){root.parent=null;},add(root){root.parent=this;}};
  const root={parent,updateMatrixWorld(){},traverse(fn){
    for(const name of ['Body','Wheel0','Wheel1','Wheel2','Wheel3'])fn({name,isMesh:true,visible:true,
      parent:root,position:{toArray:()=>[0,.27,0]},matrixWorld:{elements:identity},material,
      geometry:{attributes:{position:attribute,normal:attribute},groups:[]}});
  }};
  const sample=()=>Object.assign([0,0,0,0,0,0,0,0,1],{wheelState:Array(42).fill(0)});
  const leader={samples:[sample(),sample()],color:'#00ff00',car:{polybotReplayRoot:root,
    getCarState:()=>({}),setCarState(){},update(){}}};
  const others={samples:[sample(),sample()],color:'#7f8040'};
  const uploads=[],trails=[],sent=[],workers=[];
  const interop={apiVersion:1,ready:true,capabilities:()=>({ready:interop.ready}),registerSwarmSource:d=>(interop.description=d,1),
    updateSwarmInstances:(...a)=>uploads.push(a),updateSwarmTrails:(...a)=>trails.push(a),
    unregisterSwarmSource(){},diagnostics:()=>({})};
  const sandbox={Float32Array,Float64Array,Uint32Array,performance,structuredClone,
    Blob:class {},URL:{createObjectURL:()=>'',revokeObjectURL(){}},
    Worker:class {constructor(){workers.push(this);}postMessage(m){sent.push(m);}terminate(){}},
    ResizeObserver:class {observe(){}disconnect(){}},
    OffscreenCanvas:class {},HTMLCanvasElement:class {transferControlToOffscreen(){}},
    document:{querySelector:()=>({width:1280,height:720,after(){},getBoundingClientRect:()=>({left:0,top:0,width:1280,height:720})}),
      createElement:()=>({style:{},setAttribute(){},remove(){},transferControlToOffscreen(){return {};}}),addEventListener(){},removeEventListener(){}},
    window:{addEventListener(){},removeEventListener(){}},setTimeout:()=>1,clearTimeout(){},
    polyShadeSwarm:interop,createSwarmKernel(){},createSwarmTrails(){},polybotSwarmWorker(){}};
  sandbox.globalThis=sandbox;
  vm.runInNewContext(readFileSync(new URL('../pml-mod/0.1.49/swarm_worker_client.mjs',import.meta.url),'utf8')
    .replace('export function','function')+';globalThis.create=createPolyBotSwarmWorker;',sandbox);
  const context={scene:{updateMatrixWorld(){},traverse(){}},three:{kn4:class {}},
    getCamera:()=>({updateMatrixWorld(){},matrixWorld:{elements:identity},matrixWorldInverse:{elements:identity},projectionMatrix:{elements:identity}})};
  const failures=[],handoffs=[],ghosts=[leader,others];
  const onFailure=(r,h)=>{failures.push(r);if(h)handoffs.push(h);};
  const controller=sandbox.create(context,ghosts,leader,onFailure);
  const worker=workers[0],generation=sent[0].generation;
  const receive=m=>worker.onmessage({data:{generation,...m}});
  receive({type:'ready',outputOffsets:[32],trailOffset:80});
  return {controller,receive,uploads,trails,sent,failures,handoffs,interop,root,parent,others,sandbox,context,ghosts,leader,onFailure,workers};
}

test('renderer handoffs retain canonical replay storage without expanding native cars/sample graphs',()=>{
 const f=fixture();f.interop.ready=false;f.interop.description.onUnavailable('renderer disabled');
 assert.equal(f.others.samples,null);assert.equal(f.handoffs.length,1);
 const packed=f.handoffs[0].canonical;
 const standalone=f.sandbox.create(f.context,f.ghosts,f.leader,f.onFailure,f.handoffs[0]);
 assert.equal(f.sent.findLast(m=>m.type==='init').shared,false);
 assert.equal(f.others.samples,null);
 const init=f.sent.findLast(m=>m.type==='init');
 f.workers.at(-1).onmessage({data:{type:'ready',generation:init.generation}});
 f.interop.ready=true;standalone.frame(4,true,1,'disappear',.75);
 assert.equal(f.handoffs.length,2);assert.equal(f.handoffs[1].canonical,packed);
 const shared=f.sandbox.create(f.context,f.ghosts,f.leader,f.onFailure,f.handoffs[1]);
 assert.equal(f.sent.findLast(m=>m.type==='init').shared,true);
 assert.equal(f.others.samples,null);shared.dispose();
});

test('worker frames commit at the camera boundary and preserve all queued skid deltas',()=>{
  const f=fixture();
  const packet=new Float32Array(96);packet[0]=2;packet[28]=1;
  f.receive({type:'instances',epoch:0,time:1,buffer:packet.slice().buffer});
  f.receive({type:'instances',epoch:0,time:2,buffer:packet.slice().buffer});
  assert.equal(f.uploads.length,0);assert.equal(f.controller.renderClock(),0);
  f.controller.present();assert.equal(f.uploads.length,2);assert.equal(f.trails.length,2);
  assert.equal(f.controller.renderClock(),2);assert.equal(f.controller.isFresh(),true);
  f.controller.present();assert.equal(f.uploads.length,2);
  assert.equal(f.sent.filter(m=>m.type==='instances_recycle').length,2);
  f.controller.dispose();assert.equal(f.root.parent,f.parent);
});

test('seek rejects stale worker transforms and restores packed replay on worker failure',()=>{
  const f=fixture(),packet=new Float32Array(96);packet[0]=2;
  assert.equal(f.others.samples,null);
  f.controller.invalidate();assert.equal(f.controller.isFresh(),false);
  f.receive({type:'instances',epoch:0,time:5,buffer:packet.slice().buffer});
  f.controller.present();assert.equal(f.uploads.length,0);
  f.receive({type:'instances',epoch:1,time:.5,buffer:packet.slice().buffer});
  f.controller.present();assert.equal(f.controller.renderClock(),.5);assert.equal(f.controller.isFresh(),true);
  f.receive({type:'error',error:'injected failure'});
  assert.deepEqual(f.failures,['injected failure']);assert.equal(f.others.samples.length,2);
  assert.equal(f.others.samples[0].wheelState.length,42);assert.equal(f.root.parent,f.parent);
});

test('native material family and flat shading survive the packed geometry boundary',()=>{
  const f=fixture(),g=f.interop.description.groups[0];
  assert.equal(g.materialType,'MeshStandardMaterial');assert.equal(g.flatShading,true);
  assert.equal(g.roughness,.4);assert.equal(g.metalness,0);assert.equal(g.envMapIntensity,1);
  f.controller.dispose();
});


test('generation colours use the same byte-channel convention as native car-style uniforms',()=>{
  const f=fixture(),colours=f.sent[0].colours;
  assert.equal(colours[3],Math.fround(127/255));assert.equal(colours[4],Math.fround(128/255));
  assert.equal(colours[5],Math.fround(64/255));f.controller.dispose();
});
