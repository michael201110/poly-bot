// Self-contained functions: the bridge embeds their source in a dedicated worker.
// No game, trainer, DOM, simulator or model APIs are available in this module.
export function createSwarmKernel(replay, mounts) {
  if (!(replay?.data instanceof Float64Array) || !(replay.offsets instanceof Uint32Array) ||
      replay.offsets.length < 2 || replay.offsets.length > 501 || replay.offsets[0] !== 0 ||
      replay.data.length !== replay.offsets[replay.offsets.length-1]*53 || mounts.length !== 12)
    throw Error('invalid packed replay');
  for(let i=1;i<replay.offsets.length;i++)if(replay.offsets[i]<=replay.offsets[i-1])throw Error('empty or unordered replay');
  const stride = 53, outStride = 64, count = replay.offsets.length - 1;
  const state = new Float64Array(count * outStride);
  const indices = new Uint32Array(count);
  const angles = new Float64Array(count * 4);
  const velocities = new Float64Array(count * 4);
  const body = new Float32Array(count * 16),bodyDouble=new Float64Array(count*16);
  const wheels = new Float32Array(count * 64);
  const q = new Float64Array(4), wq = new Float64Array(4);
  let previous = -1;
  function compose(target, offset, x, y, z, a, b, c, d, sx = 1) {
    const x2 = a + a, y2 = b + b, z2 = c + c;
    const xx = a * x2, xy = a * y2, xz = a * z2;
    const yy = b * y2, yz = b * z2, zz = c * z2;
    const wx = d * x2, wy = d * y2, wz = d * z2;
    target[offset] = (1 - yy - zz) * sx;
    target[offset+1] = (xy + wz) * sx; target[offset+2] = (xz - wy) * sx; target[offset+3] = 0;
    target[offset+4] = xy - wz; target[offset+5] = 1 - xx - zz; target[offset+6] = yz + wx; target[offset+7] = 0;
    target[offset+8] = xz + wy; target[offset+9] = yz - wx; target[offset+10] = 1 - xx - yy; target[offset+11] = 0;
    target[offset+12] = x; target[offset+13] = y; target[offset+14] = z; target[offset+15] = 1;
  }
  function multiplyQuaternion(a, b, c, d, x, y, z, w, output) {
    output[0]=a*w+d*x+b*z-c*y; output[1]=b*w+d*y+c*x-a*z;
    output[2]=c*w+d*z+a*y-b*x; output[3]=d*w-a*x-b*y-c*z;
  }
  function evaluate(time, dt, reset, opacity, behavior, fade) {
    if (!Number.isFinite(time) || !Number.isFinite(dt) || time < 0 || dt < 0) throw Error('invalid replay clock');
    const data = replay.data;
    const coastDuration = behavior === 2 ? Math.max(.75, fade) : .75;
    const backwards = time < previous;
    for (let car=0; car<count; car++) {
      const begin = replay.offsets[car], end = replay.offsets[car+1]-1;
      const start = data[begin*stride+1], duration = data[end*stride+1]-start;
      const at = car*outStride, ended = time > duration;
      const coast = ended && behavior !== 1 ? Math.min(time-duration, coastDuration) : 0;
      const local = Math.min(time,duration), target = start+local;
      let index = indices[car];
      if (backwards || index <= begin || index > end) index = Math.min(end,begin+1);
      while (index < end && data[index*stride+1] < target) index++;
      indices[car] = index;
      let left = Math.max(begin,index-1)*stride, right = index*stride;
      if (target <= start || begin === end) left=right=begin*stride;
      const span = data[right+1]-data[left+1];
      const f = span > 0 ? Math.max(0,Math.min(1,(target-data[left+1])/span)) : (left===right ? 0 : 1);
      const source = left===right;
      const tail = end*stride, prior = Math.max(begin,end-1)*stride;
      const tailSpan = data[tail+1]-data[prior+1];
      const travel = coast-.25*coast*coast/coastDuration;
      for (let j=2;j<5;j++) state[at+j]=data[left+j]+f*(data[right+j]-data[left+j]) +
        (coast>0 && tailSpan>0 ? (data[tail+j]-data[prior+j])/tailSpan*travel : 0);
      let ax=data[left+5],ay=data[left+6],az=data[left+7],aw=data[left+8];
      let bx=data[right+5],by=data[right+6],bz=data[right+7],bw=data[right+8];
      let dot=ax*bx+ay*by+az*bz+aw*bw;
      if(dot<0){dot=-dot;bx=-bx;by=-by;bz=-bz;bw=-bw;}
      let l=1-f,r=f;
      if(!source && dot<=.9995){const angle=Math.acos(Math.max(-1,Math.min(1,dot))),inv=1/Math.sin(angle);
        l=Math.sin((1-f)*angle)*inv;r=Math.sin(f*angle)*inv;}
      q[0]=l*ax+r*bx;q[1]=l*ay+r*by;q[2]=l*az+r*bz;q[3]=l*aw+r*bw;
      if(!source){const n=1/Math.hypot(q[0],q[1],q[2],q[3]);for(let j=0;j<4;j++)q[j]*=n;}
      for(let j=0;j<4;j++)state[at+5+j]=q[j];
      state[at] = Math.round((data[left+1]+f*span+coast)*1000);
      state[at+1] = local;
      const nearest = f<.5 ? left : right;
      for(let wheel=0;wheel<4;wheel++) {
        const offset=9+wheel*10, current=nearest+offset;
        const smooth=data[left+offset] && data[right+offset];
        state[at+offset]=ended ? 0 : data[current];
        for(let j=1;j<7;j++)state[at+offset+j]=smooth ?
          data[left+offset+j]+f*(data[right+offset+j]-data[left+offset+j]) : data[current+j];
        const n=Math.hypot(state[at+offset+4],state[at+offset+5],state[at+offset+6])||1;
        for(let j=4;j<7;j++)state[at+offset+j]/=n;
        for(let j=7;j<10;j++)state[at+offset+j]=data[left+offset+j]+f*(data[right+offset+j]-data[left+offset+j]);
      }
      state[at+49]=data[left+49]+f*(data[right+49]-data[left+49]);
      state[at+50]=data[nearest+50];
      state[at+51]=opacity*(ended && behavior===2 ? Math.max(0,1-coast/coastDuration) : 1);
      state[at+52]=!(ended && behavior!==1 && time-duration>=coastDuration) ? 1 : 0;
      state[at+53]=span>0 ? Math.hypot(data[right+2]-data[left+2],data[right+3]-data[left+3],data[right+4]-data[left+4])/span*3.6*(ended?1-.5*coast/coastDuration:1) : 0;
      state[at+54]=ended?1:0;
      const x=state[at+2],y=state[at+3],z=state[at+4];
      compose(bodyDouble,car*16,x,y,z,q[0],q[1],q[2],q[3]);
      for(let j=0;j<16;j++)body[car*16+j]=bodyDouble[car*16+j];
      const m=car*16;
      for(let wheel=0;wheel<4;wheel++) {
        const offset=9+wheel*10, k=car*4+wheel;
        const step = reset || ended ? 0 : dt;
        const delta=state[at+offset+8];
        if(state[at+offset]) velocities[k]=1000*delta;
        else {
          const controls = source || f<1 ? left : right;
          const rear=wheel>=2, brake=data[controls+52]>0, throttle=data[controls+51]>0;
          if(brake){if(state[at+50])velocities[k]=Math.sign(velocities[k])*Math.max(0,Math.abs(velocities[k])-50*step);
            else if(rear)velocities[k]-=25*step;}
          else if(throttle && rear)velocities[k]+=50*step;
          velocities[k]*=Math.max(0,1-step);
        }
        angles[k]+=velocities[k]*step;
        const mx=mounts[wheel*3],my=mounts[wheel*3+1]-state[at+offset+7],mz=mounts[wheel*3+2];
        const px=x+bodyDouble[m]*mx+bodyDouble[m+4]*my+bodyDouble[m+8]*mz;
        const py=y+bodyDouble[m+1]*mx+bodyDouble[m+5]*my+bodyDouble[m+9]*mz;
        const pz=z+bodyDouble[m+2]*mx+bodyDouble[m+6]*my+bodyDouble[m+10]*mz;
        for(let j=0;j<4;j++)wq[j]=q[j];
        if(wheel<2)multiplyQuaternion(wq[0],wq[1],wq[2],wq[3],0,Math.sin(state[at+49]/2),0,Math.cos(state[at+49]/2),wq);
        multiplyQuaternion(wq[0],wq[1],wq[2],wq[3],Math.sin(angles[k]/2),0,0,Math.cos(angles[k]/2),wq);
        compose(wheels,k*16,px,py,pz,wq[0],wq[1],wq[2],wq[3],wheel%2 ? -1 : 1);
      }
    }
    previous=time;
    return state;
  }
  // Conservative 8 m car sphere, no distance/LOD geometry reduction. The
  // projection planes reject only wholly off-screen cars; trails are separate.
  function inFrustum(matrix, car) {
    const o=car*outStride,x=state[o+2],y=state[o+3],z=state[o+4];
    for(let axis=0;axis<3;axis++)for(let sign=-1;sign<=1;sign+=2){
      const a=matrix[3]+sign*matrix[axis],b=matrix[7]+sign*matrix[axis+4];
      const c=matrix[11]+sign*matrix[axis+8],d=matrix[15]+sign*matrix[axis+12];
      if(a*x+b*y+c*z+d < -8*Math.hypot(a,b,c))return false;
    }
    return true;
  }
  return {count,state,body,bodyDouble,wheels,angles,velocities,evaluate,inFrustum};
}

export function createSwarmTrails(kernel,mounts) {
  // Port of PolyTrack 0.6.3's native skid strip: 1000 allocated slots, wrap at
  // 999, two projected tyre edges, identical winding, threshold and cooldown.
  const count=kernel.count*4,cooldown=new Float64Array(count).fill(.075);
  const previous=new Float64Array(count*6),valid=new Uint8Array(count),rings=new Uint16Array(count);
  const effectFrames=new Float64Array(kernel.count).fill(-Infinity),corners=new Float64Array(6);
  let resetCount=0;
  function update(dt,reset,target,offset){let emitted=0;
    if(reset){valid.fill(0);rings.fill(0);effectFrames.fill(-Infinity);resetCount++;}
    const state=kernel.state,matrix=kernel.bodyDouble;
    for(let car=0;car<kernel.count;car++){
      const at=car*64,m=car*16,tick=state[at];
      const effectStep=tick>=effectFrames[car]+10;
      if(effectStep)effectFrames[car]=tick;
      for(let wheel=0;wheel<4;wheel++){
        const k=car*4+wheel,w=at+9+wheel*10;
        const slipping=state[w]&&state[w+9]<Math.pow(Math.min(1,Math.abs(state[w+8])/.08),3)*.5;
        if(effectStep){if(slipping)cooldown[k]=Math.max(0,cooldown[k]-.01);
          else {cooldown[k]=.075;valid[k]=0;}}
        if(!dt||reset||state[at+54]||!slipping||cooldown[k]!==0)continue;
        const x=mounts[wheel*3],y=mounts[wheel*3+1]-state[w+7],z=mounts[wheel*3+2];
        const px=matrix[m+12]+matrix[m]*x+matrix[m+4]*y+matrix[m+8]*z;
        const py=matrix[m+13]+matrix[m+1]*x+matrix[m+5]*y+matrix[m+9]*z;
        const pz=matrix[m+14]+matrix[m+2]*x+matrix[m+6]*y+matrix[m+10]*z;
        const nx=state[w+4],ny=state[w+5],nz=state[w+6];
        for(let side=0;side<2;side++){
          const dx=side?-.172:.172;
          let cx=px+matrix[m]*dx-matrix[m+4]*.3,cy=py+matrix[m+1]*dx-matrix[m+5]*.3,cz=pz+matrix[m+2]*dx-matrix[m+6]*.3;
          const distance=(cx-state[w+1])*nx+(cy-state[w+2])*ny+(cz-state[w+3])*nz;
          cx-=nx*distance;cy-=ny*distance;cz-=nz*distance;
          corners[side*3]=cx;corners[side*3+1]=cy;corners[side*3+2]=cz;
        }
        const old=k*6;
        if(valid[k]){const o=offset+emitted*13;target[o]=k*999+rings[k];
          for(let j=0;j<3;j++){target[o+1+j]=corners[j];target[o+4+j]=previous[old+j];target[o+7+j]=corners[3+j];target[o+10+j]=previous[old+3+j];}
          rings[k]=(rings[k]+1)%999;emitted++;}
        for(let j=0;j<6;j++)previous[old+j]=corners[j];valid[k]=1;
      }
    }
    return emitted;
  }
  return {update,cooldown,rings,previous,valid,get resetCount(){return resetCount;}};
}

export function polybotSwarmWorker() {
  let gl, canvas, kernel, groups=[], depth, packet=null, packetView=null, generation=0, disposed=false;
  let colors, opacity=1, behavior=2, fade=.75, lastTime=-1, resetId=-1;
  let lastStats=0, frames=0, cursor=0, gpuMs=0, gpuPending=[];
  const history=new Float64Array(240*4), vp=new Float32Array(16), sun=new Float32Array(3);
  const ambient=new Float32Array(3), light=new Float32Array(3), scratch=new Float32Array(16), projection=new Float32Array(16);
  const visible=new Uint8Array(500);
  let extension, program, u, vaoDepth, depthProgram, depthLocation, depthCount=0;
  let shared=false,outputPool=[],outputOffsets=[],outputLength=32,lastOutput=-1,trails,trailOffset=0;
  const vs=`#version 300 es
  precision highp float;
  layout(location=0) in vec3 position;layout(location=1) in vec3 normal;
  layout(location=2) in mat4 instance;layout(location=6) in vec4 colour;
  layout(location=7) in vec4 parameters;
  layout(location=8) in vec3 morph0;layout(location=9) in vec3 morph1;
  layout(location=10) in vec3 morph2;layout(location=11) in vec3 morph3;
  uniform mat4 viewProjection;uniform bool morph;out vec3 n;out vec3 world;
  out vec4 paint;out float brake;
  void main(){vec3 p=position;if(morph)p+=morph0*parameters.x+morph1*parameters.y+morph2*parameters.z+morph3*parameters.w;
    vec4 w=instance*vec4(p,1.);world=w.xyz;n=normalize(mat3(instance)*normal);paint=colour;brake=parameters.x;
    gl_Position=viewProjection*w;}`;
  const fs=`#version 300 es
  precision highp float;in vec3 n;in vec3 world;in vec4 paint;in float brake;
  uniform vec3 baseColour;uniform vec3 sunDirection;uniform vec3 sunColour;uniform vec3 ambientColour;
  uniform vec3 fogColour;uniform vec3 eye;uniform float exposure;uniform float fogStrength;
  uniform int paintMode;out vec4 outColour;
  void main(){vec3 norm=normalize(n);if(!gl_FrontFacing)norm=-norm;
    vec3 base=paintMode==1?paint.rgb:baseColour;
    vec3 lit=base*(ambientColour+sunColour*max(0.,dot(norm,sunDirection)));
    if(paintMode==2)lit=mix(vec3(.03,0.,0.),vec3(1.,.02,.005),brake);
    vec3 h=normalize(sunDirection+normalize(eye-world));
    if(paintMode!=3)lit+=sunColour*pow(max(0.,dot(norm,h)),48.)*.14;
    lit*=exposure;lit=lit/(vec3(1.)+lit*.22);
    lit=mix(lit,fogColour,clamp((length(eye-world)-80.)*.0013*fogStrength,0.,.7));
    lit=mix(12.92*lit,1.055*pow(max(lit,vec3(0.)),vec3(1./2.4))-.055,step(vec3(.0031308),lit));
    outColour=vec4(lit,paint.a);}`;
  function shader(type,source){const s=gl.createShader(type);gl.shaderSource(s,source);gl.compileShader(s);
    if(!gl.getShaderParameter(s,gl.COMPILE_STATUS))throw Error(gl.getShaderInfoLog(s));return s;}
  function makeProgram(vertex,fragment){const p=gl.createProgram(),v=shader(gl.VERTEX_SHADER,vertex),f=shader(gl.FRAGMENT_SHADER,fragment);
    gl.attachShader(p,v);gl.attachShader(p,f);gl.linkProgram(p);gl.deleteShader(v);gl.deleteShader(f);
    if(!gl.getProgramParameter(p,gl.LINK_STATUS))throw Error(gl.getProgramInfoLog(p));return p;}
  function attribute(location,array,size,divisor=0,stride=0,offset=0,buffer=null){
    const b=buffer||gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,b);
    if(array)gl.bufferData(gl.ARRAY_BUFFER,array,divisor?gl.DYNAMIC_DRAW:gl.STATIC_DRAW);
    gl.enableVertexAttribArray(location);gl.vertexAttribPointer(location,size,gl.FLOAT,false,stride,offset);
    gl.vertexAttribDivisor(location,divisor);return b;
  }
  function setupGroup(g,count){g.vao=gl.createVertexArray();gl.bindVertexArray(g.vao);g.buffers=[];
    g.buffers.push(attribute(0,g.positions,3));g.buffers.push(attribute(1,g.normals,3));
    for(let j=0;j<4;j++)if(g.morphs?.[j])g.buffers.push(attribute(8+j,g.morphs[j],3));
    g.data=new Float32Array(count*(g.kind===1?4:1)*24);
    g.instance=attribute(2,g.data,4,1,96,0);g.buffers.push(g.instance);
    for(let j=1;j<4;j++)attribute(2+j,null,4,1,96,j*16,g.instance);
    attribute(6,null,4,1,96,64,g.instance);attribute(7,null,4,1,96,80,g.instance);
    g.vertices=g.positions.length/3;g.positions=null;g.normals=null;g.morphs=null;return g;
  }
  function init(message){canvas=message.canvas;generation=message.generation;
    shared=message.shared===true;
    if(shared){
      kernel=createSwarmKernel(message.replay,message.mounts);colors=message.colours;
      groups=message.groups;trails=createSwarmTrails(kernel,message.mounts);
      for(const g of groups){outputOffsets.push(outputLength);outputLength+=kernel.count*(g.kind===1?4:1)*24;}
      trailOffset=outputLength;outputLength+=kernel.count*4*13;
      outputPool=[new Float32Array(outputLength),new Float32Array(outputLength),new Float32Array(outputLength)];
      postMessage({type:'ready',generation,backend:'Worker + PolyShade WebGPU',replayBytes:message.replay.data.byteLength,
        geometryBytes:message.geometryBytes,batches:groups.length,outputOffsets,outputLength,trailOffset});return;
    }
    gl=canvas.getContext('webgl2',{alpha:true,premultipliedAlpha:false,antialias:true,depth:true,preserveDrawingBuffer:false});
    if(!gl)throw Error('worker WebGL2 unavailable');
    extension=gl.getExtension('EXT_disjoint_timer_query_webgl2');
    kernel=createSwarmKernel(message.replay,message.mounts);colors=message.colours;
    program=makeProgram(vs,fs);u={};
    for(const name of ['viewProjection','morph','baseColour','sunDirection','sunColour','ambientColour','fogColour','eye','exposure','fogStrength','paintMode'])u[name]=gl.getUniformLocation(program,name);
    groups=message.groups.map(g=>setupGroup(g,kernel.count));
    depthProgram=makeProgram('#version 300 es\nlayout(location=0) in vec3 position;uniform mat4 vp;void main(){gl_Position=vp*vec4(position,1.);}',
      '#version 300 es\nprecision highp float;out vec4 colour;void main(){colour=vec4(0.);}');
    depthLocation=gl.getUniformLocation(depthProgram,'vp');vaoDepth=gl.createVertexArray();gl.bindVertexArray(vaoDepth);
    depth=attribute(0,message.occluders,3);depthCount=message.occluders.length/3;
    gl.bindVertexArray(null);postMessage({type:'ready',generation,backend:'Worker WebGL2',replayBytes:message.replay.data.byteLength,
      geometryBytes:message.geometryBytes,proxyTriangles:depthCount/3,batches:groups.length});
    requestAnimationFrame(frame);
  }
  function multiply(a,b,out){for(let c=0;c<4;c++)for(let r=0;r<4;r++)out[c*4+r]=a[r]*b[c*4]+a[4+r]*b[c*4+1]+a[8+r]*b[c*4+2]+a[12+r]*b[c*4+3];}
  function sharedFrame(){if(!packetView||!outputPool.length||packetView[57]===lastOutput)return;
    const start=performance.now(),p=packetView,reset=p[2]!==resetId,time=p[0];
    const dt=reset||lastTime<0?0:Math.max(0,time-lastTime);
    const state=kernel.evaluate(time,dt,reset,p[3],p[4],p[5]);
    lastOutput=p[57];resetId=p[2];lastTime=time;
    const interpolationEnd=performance.now(),target=outputPool.pop();
    for(let j=0;j<16;j++){scratch[j]=p[8+j];projection[j]=p[24+j];}multiply(projection,scratch,vp);
    let visibleCount=0;
    for(let c=0;c<kernel.count;c++){visible[c]=state[c*64+52]&&kernel.inFrustum(vp,c)?1:0;visibleCount+=visible[c];}
    for(let index=0;index<groups.length;index++){
      const g=groups[index],iterations=g.kind===1?4:1;target[index]=kernel.count*iterations;
      for(let car=0;car<kernel.count;car++)for(let wheel=0;wheel<iterations;wheel++){
        const input=g.kind===1?kernel.wheels:kernel.body,at=(g.kind===1?car*4+wheel:car)*16;
        const o=outputOffsets[index]+(car*iterations+wheel)*24,s=car*64;
        for(let j=0;j<16;j++)target[o+j]=input[at+j];
        // Stable slots preserve exact previous-instance identity for temporal
        // motion vectors. Reject only wholly off-screen / despawned geometry.
        if(!visible[car])target[o+12]=target[o+13]=target[o+14]=1e8;
        for(let j=0;j<3;j++)target[o+16+j]=colors[car*3+j];target[o+19]=state[s+51];
        if(g.kind===2)for(let j=0;j<4;j++)target[o+20+j]=2*state[s+9+j*10+7]+.06;
        else target[o+20]=state[s+50];
      }
    }
    target[28]=trails.update(dt,reset,target,trailOffset);target[29]=reset?1:0;
    const elapsed=performance.now()-start;history[cursor*4]=elapsed;cursor=(cursor+1)%240;frames++;
    postMessage({type:'instances',generation,buffer:target.buffer,time,epoch:resetId},[target.buffer]);
    const now=performance.now();if(now-lastStats>=500){lastStats=now;
      const samples=Math.min(frames,240),times=Array.from({length:samples},(_,i)=>history[i*4]).sort((a,b)=>a-b),positions=new Float64Array(kernel.count*4);
      for(let c=0;c<kernel.count;c++){const at=c*64;positions[c*4]=state[at+2];positions[c*4+1]=state[at+3];positions[c*4+2]=state[at+4];positions[c*4+3]=state[at+52];}
      postMessage({type:'stats',generation,positions,stats:{backend:'Worker + PolyShade WebGPU',cars:kernel.count,visible:visibleCount,lod:{full:visibleCount,medium:0,far:0},
        workerMedianMs:times[times.length>>1],workerP95Ms:times[Math.floor(times.length*.95)],workerP99Ms:times[Math.floor(times.length*.99)],
        interpolationMs:interpolationEnd-start,packingMs:elapsed-(interpolationEnd-start),frames,clock:time}},[positions.buffer]);
    }
  }
  function frame(now){if(disposed)return;const begin=performance.now();
    if(packet){const p=packetView,time=p[0],reset=p[2]!==resetId;
      const dt=reset||lastTime<0?0:Math.max(0,time-lastTime);resetId=p[2];lastTime=time;
      opacity=p[3];behavior=p[4];fade=p[5];
      const state=kernel.evaluate(time,dt,reset,opacity,behavior,fade),lookupEnd=performance.now();
      for(let j=0;j<16;j++){scratch[j]=p[8+j];projection[j]=p[24+j];}
      multiply(projection,scratch,vp);
      if(canvas.width!==p[6]||canvas.height!==p[7]){canvas.width=p[6];canvas.height=p[7];}
      gl.viewport(0,0,canvas.width,canvas.height);gl.clearColor(0,0,0,0);gl.colorMask(true,true,true,true);gl.depthMask(true);
      gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);gl.enable(gl.DEPTH_TEST);gl.depthFunc(gl.LEQUAL);gl.disable(gl.CULL_FACE);
      let query=null;
      if(extension && gpuPending.length<6){query=gl.createQuery();gl.beginQuery(extension.TIME_ELAPSED_EXT,query);}
      gl.useProgram(depthProgram);gl.uniformMatrix4fv(depthLocation,false,vp);gl.bindVertexArray(vaoDepth);
      gl.colorMask(false,false,false,false);gl.drawArrays(gl.TRIANGLES,0,depthCount);gl.colorMask(true,true,true,true);
      gl.useProgram(program);gl.uniformMatrix4fv(u.viewProjection,false,vp);
      for(let j=0;j<3;j++){sun[j]=p[40+j];light[j]=p[43+j];ambient[j]=p[46+j];}
      gl.uniform3fv(u.sunDirection,sun);gl.uniform3fv(u.sunColour,light);gl.uniform3fv(u.ambientColour,ambient);
      gl.uniform3f(u.eye,p[49],p[50],p[51]);gl.uniform3f(u.fogColour,p[52],p[53],p[54]);
      gl.uniform1f(u.exposure,p[55]);gl.uniform1f(u.fogStrength,p[56]);
      let visibleCount=0,draws=0,triangles=0;
      for(let car=0;car<kernel.count;car++){visible[car]=state[car*64+52]&&kernel.inFrustum(vp,car)?1:0;visibleCount+=visible[car];}
      const matrixBegin=performance.now();
      for(const g of groups){let instances=0;
        for(let car=0;car<kernel.count;car++){if(!visible[car])continue;const s=car*64;
          const iterations=g.kind===1?4:1;
          for(let wheel=0;wheel<iterations;wheel++){
            const m=g.kind===1?kernel.wheels:kernel.body,at=g.kind===1?(car*4+wheel)*16:car*16,o=instances*24;
            for(let j=0;j<16;j++)g.data[o+j]=m[at+j];
            for(let j=0;j<3;j++)g.data[o+16+j]=colors[car*3+j];g.data[o+19]=state[s+51];
            if(g.kind===2)for(let j=0;j<4;j++)g.data[o+20+j]=2*state[s+9+j*10+7]+.06;
            else g.data[o+20]=state[s+50];
            instances++;
          }
        }
        if(!instances)continue;
        gl.bindVertexArray(g.vao);gl.bindBuffer(gl.ARRAY_BUFFER,g.instance);
        gl.bufferSubData(gl.ARRAY_BUFFER,0,g.data,0,instances*24);
        gl.uniform1i(u.morph,g.kind===2);gl.uniform1i(u.paintMode,g.paintMode);
        gl.uniform3fv(u.baseColour,g.colour);
        gl.enable(gl.BLEND);gl.blendFunc(gl.SRC_ALPHA,gl.ONE_MINUS_SRC_ALPHA);
        gl.depthMask(opacity===1);gl.drawArraysInstanced(gl.TRIANGLES,0,g.vertices,instances);
        draws++;triangles+=g.vertices/3*instances;
      }
      const submission=performance.now()-matrixBegin;
      if(query){gl.endQuery(extension.TIME_ELAPSED_EXT);gpuPending.push(query);}
      while(gpuPending.length&&gl.getQueryParameter(gpuPending[0],gl.QUERY_RESULT_AVAILABLE)){
        const q=gpuPending.shift();if(!gl.getParameter(extension.GPU_DISJOINT_EXT))gpuMs=gl.getQueryParameter(q,gl.QUERY_RESULT)/1e6;gl.deleteQuery(q);}
      const elapsed=performance.now()-begin;
      history[cursor*4]=elapsed;history[cursor*4+1]=lookupEnd-begin;history[cursor*4+2]=submission;history[cursor*4+3]=gpuMs;
      cursor=(cursor+1)%240;frames++;
      if(now-lastStats>=500){lastStats=now;const positions=new Float64Array(kernel.count*4);
        for(let c=0;c<kernel.count;c++){const s=c*64;positions[c*4]=state[s+2];positions[c*4+1]=state[s+3];positions[c*4+2]=state[s+4];positions[c*4+3]=state[s+52];}
        const samples=Math.min(frames,240),times=Array.from({length:samples},(_,i)=>history[i*4]).sort((a,b)=>a-b);
        postMessage({type:'stats',generation,positions,stats:{backend:'Worker WebGL2',cars:kernel.count,visible:visibleCount,
          lod:{full:visibleCount,medium:0,far:0},draws,triangles,proxyDraws:1,proxyTriangles:depthCount/3,
          workerMedianMs:times[times.length>>1],workerP95Ms:times[Math.floor(times.length*.95)],
          workerP99Ms:times[Math.floor(times.length*.99)],gpuMs,interpolationMs:lookupEnd-begin,
          submissionMs:submission,frames,clock:time}},[positions.buffer]);
      }
    }
    requestAnimationFrame(frame);
  }
  self.onmessage=event=>{try{const m=event.data;
    if(m.type==='init'){init(m);return;}
    if(m.generation!==generation)return;
    if(m.type==='frame'){if(!(m.buffer instanceof ArrayBuffer)||m.buffer.byteLength!==64*8)throw Error('invalid frame packet');
      if(packet)postMessage({type:'recycle',generation,buffer:packet},[packet]);packet=m.buffer;packetView=new Float64Array(packet);if(shared)sharedFrame();}
    else if(m.type==='instances_recycle'){if(!(m.buffer instanceof ArrayBuffer)||m.buffer.byteLength!==outputLength*4||outputPool.length>=3)throw Error('invalid instance recycle');
      outputPool.push(new Float32Array(m.buffer));if(shared)sharedFrame();}
    else if(m.type==='colours'){if(!(m.colours instanceof Float32Array)||m.colours.length!==kernel.count*3)throw Error('invalid colours');colors=m.colours;}
    else if(m.type==='dispose'){disposed=true;if(shared){kernel=null;groups=[];packet=null;outputPool=[];close();return;}
      for(const g of groups){for(const b of g.buffers)gl.deleteBuffer(b);gl.deleteVertexArray(g.vao);}
      gl.deleteBuffer(depth);gl.deleteVertexArray(vaoDepth);gl.deleteProgram(program);gl.deleteProgram(depthProgram);
      for(const q of gpuPending)gl.deleteQuery(q);gl.getExtension('WEBGL_lose_context')?.loseContext();kernel=null;packet=null;groups=[];close();}
    else throw Error('unknown swarm worker message');
  }catch(error){postMessage({type:'error',generation,error:String(error?.message||error)});}};
}
