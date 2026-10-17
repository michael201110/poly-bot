import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import vm from 'node:vm';
import {createSwarmKernel,polybotSwarmWorker} from '../pml-mod/0.1.48/swarm_worker.mjs';
const fixture=JSON.parse(readFileSync(new URL('fixtures/swarm-recorded.json',import.meta.url)));
const mounts=new Float64Array([.627909,.27,1.3478,-.627909,.27,1.3478,.720832,.27,-1.52686,-.720832,.27,-1.52686]);
const old=readFileSync(new URL('../pml-mod/0.1.47/replay_renderer.mjs',import.meta.url),'utf8')
  .replace('export function installPolyBotReplayRenderer','function installPolyBotReplayRenderer')
  .replace('globalThis.__polybotStopVisualReplay =','globalThis.reference={interpolate:replayInterpolateInto,slerp:replaySlerpInto,apply:replayApplyWheels,validate:replayValidateSample,reset:replayResetIndex}; globalThis.__polybotStopVisualReplay =');
const reference={performance,TextEncoder,structuredClone};reference.globalThis=reference;
vm.runInNewContext(old+';installPolyBotReplayRenderer();',reference);
function replay(count=1){
  const samples=fixture.samples.map((row,i)=>{const sample=reference.reference.validate([...row],null);sample.wheelState=fixture.wheels[i];return sample;});
  const data=new Float64Array(samples.length*count*53),offsets=new Uint32Array(count+1);
  for(let c=0;c<count;c++){offsets[c]=c*samples.length;
    samples.forEach((s,i)=>{const at=(c*samples.length+i)*53;data.set(s,at);data.set(s.wheelState,at+9);});}
  offsets[count]=count*samples.length;return {data,offsets,samples};
}
const close=(a,b,tolerance=1e-10)=>assert.ok(Math.abs(a-b)<=tolerance,`${a} != ${b}`);
test('recorded timestamps, quaternion SLERP, wheel telemetry and coast match native equations',()=>{
  const recording=replay(),kernel=createSwarmKernel(recording,mounts);
  const ghost={samples:recording.samples,sampleIndex:1};let previous=0;
  const duration=ghost.samples.at(-1)[1]-ghost.samples[0][1];
  for(const time of [0,.01,.05,.123,.5,1,2,duration,duration+.2,duration+.749,.5,0]){
    if(time<previous)reference.reference.reset(ghost,time+ghost.samples[0][1]);
    const state=kernel.evaluate(time,Math.max(0,time-previous),time<previous,1,2,.75);
    const interval={};reference.reference.interpolate(ghost,Math.min(time,duration),interval);
    const {left,right,fraction:f,source}=interval,q={};
    if(source)Object.assign(q,{x:source[5],y:source[6],z:source[7],w:source[8]});else reference.reference.slerp(left,right,f,q);
    for(const [i,key]of ['x','y','z','w'].entries())close(state[5+i],q[key]);
    const coast=Math.max(0,Math.min(.75,time-duration)),tail=ghost.samples.at(-1),prior=ghost.samples.at(-2);
    for(let j=2;j<5;j++)close(state[j],(source?source[j]:left[j]+f*(right[j]-left[j]))+
      (coast>0?(tail[j]-prior[j])/(tail[1]-prior[1])*(coast-.25*coast*coast/.75):0));
    const wheels={wheelContact:[null,null,null,null],wheelSuspensionLength:[],wheelDeltaRotation:[],wheelSkidInfo:[]};
    reference.reference.apply(wheels,left.wheelState,right.wheelState,f);
    for(let i=0;i<4;i++){
      close(state[9+i*10+7],wheels.wheelSuspensionLength[i]);close(state[9+i*10+8],wheels.wheelDeltaRotation[i]);close(state[9+i*10+9],wheels.wheelSkidInfo[i]);
      if(time<=duration&&wheels.wheelContact[i])for(let j=0;j<3;j++)close(state[9+i*10+1+j],wheels.wheelContact[i].position[['x','y','z'][j]]);
    }
    close(state[49],wheels.steering);assert.equal(state[50]===1,wheels.brakeLightEnabled);
    close(state[51],time>duration?1-coast/.75:1);previous=time;
  }
});
test('packed buffers stay contiguous and retain identity for 1..500 cars',()=>{
  for(const count of [1,25,50,100,250,500]){
    const kernel=createSwarmKernel(replay(count),mounts),body=kernel.body,wheels=kernel.wheels,state=kernel.state;
    kernel.evaluate(.1,.1,false,1,0,.75);kernel.evaluate(.2,.1,false,1,0,.75);
    assert.equal(kernel.body,body);assert.equal(kernel.wheels,wheels);assert.equal(kernel.state,state);
    assert.equal(body.length,count*16);assert.equal(wheels.length,count*64);
    assert.equal(state[(count-1)*64+52],1);assert.ok(wheels.every(Number.isFinite));
  }
});
test('spawn/despawn, seek, freeze and fade timing preserve endpoint behavior',()=>{
  const recording=replay(),k=createSwarmKernel(recording,mounts),duration=recording.samples.at(-1)[1];
  k.evaluate(duration+.749,0,false,.7,2,.75);assert.equal(k.state[52],1);
  k.evaluate(duration+.75,0,false,.7,2,.75);assert.equal(k.state[52],0);
  k.evaluate(duration+10,0,false,.7,1,.75);assert.equal(k.state[52],1);close(k.state[51],.7);
  k.evaluate(0,0,true,.7,2,.75);close(k.state[2],recording.samples[0][2]);assert.equal(k.state[52],1);
});
test('conservative frustum rejection never drops intersecting car bounds',()=>{
  const k=createSwarmKernel(replay(),mounts),identity=new Float32Array([1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1]);
  k.state[2]=k.state[3]=k.state[4]=0;assert.equal(k.inFrustum(identity,0),true);
  k.state[2]=8;assert.equal(k.inFrustum(identity,0),true);
  k.state[2]=20;assert.equal(k.inFrustum(identity,0),false);
});
test('worker protocol rejects malformed packets and cannot access simulation APIs',()=>{
  const messages=[],self={},context={self,postMessage:m=>messages.push(m),Float32Array,Float64Array,Uint32Array,Uint8Array,ArrayBuffer,performance,close(){}};
  vm.runInNewContext(createSwarmKernel.toString()+';('+polybotSwarmWorker.toString()+')();',context);
  self.onmessage({data:{type:'init',generation:1,shared:true,replay:replay(),mounts,colours:new Float32Array([0,1,0]),groups:[{kind:0}],geometryBytes:36}});
  assert.equal(messages.at(-1).type,'ready');
  self.onmessage({data:{type:'frame',generation:1,buffer:new ArrayBuffer(2)}});assert.equal(messages.at(-1).type,'error');
  self.onmessage({data:{type:'physics_step',generation:1}});assert.equal(messages.at(-1).type,'error');
  self.onmessage({data:{type:'dispose',generation:1}});
});
