// GPU ports of LightingEngine::filter_shadows and PostProcessor::resolve_frame.
// History stores linear HDR RGB; alpha is the native history-used/geometry mask.
const glPostBindings = `#version 300 es
precision highp float;
uniform sampler2D positionTex, normalTex, sourceTex, historyTex;
uniform sampler2D previousPositionTex, previousNormalTex;
uniform vec2 previousJitter;
uniform bool cameraTranslated;
uniform mat4 view, previousVP;
uniform vec2 size, jitter;
uniform vec4 taaParams, filterParams; // blend, clamp, history-valid, sharpen / center, neighbor, depth, normal
uniform bool taaEnabled, filterEnabled;
uniform float exposure;
out vec4 color;
ivec2 bounded(ivec2 p) { return clamp(p,ivec2(0),ivec2(size)-1); }
vec3 sourceAt(ivec2 p) { return texelFetch(sourceTex,bounded(p),0).rgb; }
`;
const glShadowFilter = glPostBindings + `
void main() {
  ivec2 p=ivec2(gl_FragCoord.xy);
  vec4 raw=texelFetch(sourceTex,p,0), world=texelFetch(positionTex,p,0);
  vec3 normal=texelFetch(normalTex,p,0).xyz;
  if(!filterEnabled || world.w==0.0 || dot(normal,normal)<=1e-6) { color=raw; return; }
  float depth=(view*vec4(world.xyz,1)).z;
  vec2 sum=raw.rg*filterParams.x;
  float weight=filterParams.x;
  const ivec2 offsets[4]=ivec2[4](ivec2(-1,0),ivec2(1,0),ivec2(0,-1),ivec2(0,1));
  for(int i=0;i<4;i++) {
    ivec2 q=p+offsets[i];
    if(any(lessThan(q,ivec2(0)))||any(greaterThanEqual(q,ivec2(size)))) continue;
    vec4 neighbor=texelFetch(positionTex,q,0);
    vec3 n=texelFetch(normalTex,q,0).xyz;
    if(neighbor.w==0.0 || dot(n,n)<=1e-6) continue;
    if(abs((view*vec4(neighbor.xyz,1)).z-depth)>filterParams.z) continue;
    if(clamp(dot(normal,n),-1.0,1.0)<filterParams.w) continue;
    sum+=texelFetch(sourceTex,q,0).rg*filterParams.y; weight+=filterParams.y;
  }
  color=vec4(weight<=1.1920929e-7?raw.rg:clamp(sum/weight,0.0,1.0),0,1);
}`;
const glTaaResolve = glPostBindings + `
bool historySurfaceMatches(ivec2 q,vec4 world,vec3 normal) {
  if(!cameraTranslated) return true;
  vec4 old=texelFetch(previousPositionTex,q,0);
  if(world.w==0.0) return old.w==0.0;
  if(old.w!=world.w) return false;
  vec3 oldNormal=texelFetch(previousNormalTex,q,0).xyz;
  // Voxel faces are planar. View-depth-relative tolerances allow distant
  // foreground trim to bleed onto the wall behind it, so test its world plane.
  return dot(oldNormal,normal)>0.95 && abs(dot(old.xyz-world.xyz,normal))<0.01;
}
bool historyVisible(vec2 screen,vec4 world,vec3 normal) {
  vec2 p=clamp(screen-0.5,vec2(0),size-1.0),f=fract(p);
  ivec2 lo=ivec2(floor(p)),hi=min(lo+1,ivec2(size)-1);
  for(int y=0;y<2;y++) for(int x=0;x<2;x++) {
    ivec2 q=ivec2(x==0?lo.x:hi.x,y==0?lo.y:hi.y);
    float w=(x==0?1.0-f.x:f.x)*(y==0?1.0-f.y:f.y);
    if(w>1e-5 && historySurfaceMatches(q,world,normal)) return true;
  }
  return false;
}
bool bilinearHistory(vec2 screen,vec4 world,vec3 normal,out vec3 value) {
  vec2 p=clamp(screen-0.5,vec2(0),size-1.0);
  ivec2 lo=ivec2(floor(p)),hi=min(lo+1,ivec2(size)-1);vec2 f=fract(p);
  vec3 sum=vec3(0);float weight=0.0;
  for(int y=0;y<2;y++) for(int x=0;x<2;x++) {
    ivec2 q=ivec2(x==0?lo.x:hi.x,y==0?lo.y:hi.y);
    float w=(x==0?1.0-f.x:f.x)*(y==0?1.0-f.y:f.y);
    if(w>0.0 && historySurfaceMatches(q,world,normal)) {sum+=texelFetch(historyTex,q,0).rgb*w;weight+=w;}
  }
  value=weight>1e-5?sum/weight:vec3(0);
  return weight>1e-5;
}
void main() {
  ivec2 p=ivec2(gl_FragCoord.xy);
  vec4 world=texelFetch(positionTex,p,0);
  bool sky=world.w==0.0;
  vec3 current=sourceAt(p), result=current;
  bool valid=taaEnabled && taaParams.z>0.5;
  vec3 previous=current;float motion=0.0;
  if(valid) {
    vec3 normal=texelFetch(normalTex,p,0).xyz;
    if(sky) valid=bilinearHistory(vec2(p)+0.5,world,normal,previous);
    else {
      vec4 clip=previousVP*vec4(world.xyz,1);
      valid=clip.w>0.05 && !any(isnan(clip)) && !any(isinf(clip));
      if(valid) {
        vec2 projected=(clip.xy/clip.w*0.5+0.5)*size;
        // Resolved color is accumulated on a stable pixel grid. Geometry is a
        // raw jittered G-buffer: the two histories have different coordinates.
        vec2 screen=projected-jitter,geometryScreen=projected-previousJitter;
        valid=all(greaterThanEqual(screen,vec2(0)))&&all(lessThanEqual(screen,size));
        if(valid && cameraTranslated) valid=historyVisible(vec2(geometryScreen.x,size.y-geometryScreen.y),world,normal);
        if(valid) {
          vec2 historyScreen=vec2(screen.x,size.y-screen.y);
          motion=length(historyScreen-(vec2(p)+0.5));
          valid=bilinearHistory(historyScreen,world,normal,previous);
        }
      }
    }
  }
  if(valid && taaParams.y>0.5) {
    vec3 lo=current, hi=current;
    for(int y=-1;y<=1;y++) for(int x=-1;x<=1;x++) {
      vec3 neighbor=sourceAt(p+ivec2(x,y));lo=min(lo,neighbor);hi=max(hi,neighbor);
    }
    previous=clamp(previous,lo,hi);
  }
  float blend=cameraTranslated?max(taaParams.x,min(0.25,motion*0.1)):taaParams.x;
  if(valid) result=previous+(current-previous)*blend;
  color=vec4(result,valid&&!sky?1.0:0.0);
}`;
// Soft threshold in linear HDR, then a normalized separable Gaussian. Extract
// BEFORE blurring so a small flame cannot become twelve displaced bright copies.
const glBloomExtract = glPostBindings + `
vec3 bright(vec3 c) {
  float l=max(c.r,max(c.g,c.b));
  float knee=clamp(l-0.75,0.0,1.5);
  return c*(max(l-1.5,knee*knee/3.0)/max(l,1e-5));
}
void main() {
  ivec2 p=ivec2(gl_FragCoord.xy)*2;
  p.y-=int(size.y)&1;
  vec3 sum=vec3(0);
  for(int y=0;y<2;y++) for(int x=0;x<2;x++) sum+=bright(sourceAt(p+ivec2(x,y)));
  color=vec4(sum*0.25,1);
}`;
function glBloomBlur(horizontal) {
  return glPostBindings + `
void main() {
  ivec2 p=ivec2(gl_FragCoord.xy),dims=textureSize(sourceTex,0);
  float sigma=clamp(size.y/240.0,0.65,4.5),weight=0.0;
  vec3 sum=vec3(0);
  for(int i=-14;i<=14;i++) {
    if(abs(float(i))>ceil(sigma*3.0)) continue;
    float w=exp(-0.5*float(i*i)/(sigma*sigma));
    ivec2 q=clamp(p+ivec2(${horizontal?'i,0':'0,i'}),ivec2(0),dims-1);
    sum+=texelFetch(sourceTex,q,0).rgb*w;weight+=w;
  }
  color=vec4(sum/weight,1);
}`;
}
const glPostPresent = glPostBindings + `
uniform bool cathedral;
uniform sampler2D bloomTex;
vec3 bloomAt(ivec2 pixel) {
  vec2 q=(vec2(pixel)+0.5)*0.5-0.5;
  q.y+=float(int(size.y)&1)*0.5;
  ivec2 lo=ivec2(floor(q)),hi=lo+1,dims=textureSize(bloomTex,0);
  vec2 f=fract(q);lo=clamp(lo,ivec2(0),dims-1);hi=clamp(hi,ivec2(0),dims-1);
  return mix(mix(texelFetch(bloomTex,lo,0).rgb,texelFetch(bloomTex,ivec2(hi.x,lo.y),0).rgb,f.x),
    mix(texelFetch(bloomTex,ivec2(lo.x,hi.y),0).rgb,texelFetch(bloomTex,hi,0).rgb,f.x),f.y);
}
void main() {
  ivec2 p=ivec2(gl_FragCoord.xy);
  vec4 resolved=texelFetch(sourceTex,p,0);
  vec3 result=resolved.rgb;
  if(taaParams.w>0.0 && resolved.a>0.5) {
    vec3 blur=(result*4.0+sourceAt(p+ivec2(-1,0))+sourceAt(p+ivec2(1,0))
                +sourceAt(p+ivec2(0,-1))+sourceAt(p+ivec2(0,1)))/8.0;
    result=max(vec3(0),result+(result-blur)*taaParams.w);
  }
  if(cathedral) result+=bloomAt(p)*0.12;
  result=max(result*max(exposure,0.0),vec3(0));result=result/(1.0+result);
  result=mix(12.92*result,1.055*pow(result,vec3(1.0/2.4))-0.055,step(vec3(0.0031308),result));
  if(texelFetch(positionTex,p,0).w!=0.0) {
    const int bayer[16]=int[16](0,8,2,10,12,4,14,6,3,11,1,9,15,7,13,5);
    int row=(int(size.y)-1-p.y)&3;
    result+=(float(bayer[row*4+(p.x&3)])-7.5)*(2.0/16.0)/255.0;
  }
  color=vec4(clamp(result,0.0,1.0),1);
}`;

function webgpuPostShader(uniforms, stage) {
  const bindings = uniforms + `
@group(0) @binding(1) var positions: texture_2d<f32>;
@group(0) @binding(2) var normals: texture_2d<f32>;
@group(0) @binding(3) var source: texture_2d<f32>;
@group(0) @binding(4) var history: texture_2d<f32>;
${stage === 'present' ? '@group(0) @binding(6) var bloom: texture_2d<f32>;' : ''}
${stage === 'taa' ? '@group(0) @binding(7) var previousPositions: texture_2d<f32>;\n@group(0) @binding(8) var previousNormals: texture_2d<f32>;' : ''}
@group(0) @binding(5) var destination: texture_storage_2d<${stage === 'present' ? 'rgba8unorm' : 'rgba32float'}, write>;
fn bounded(p: vec2i) -> vec2i { return clamp(p,vec2i(0),vec2i(u.params.xy)-1); }
fn sourceAt(p: vec2i) -> vec3f { return textureLoad(source,bounded(p),0).xyz; }
`;
  const filter = `
fn process(p: vec2i) -> vec4f {
  let raw=textureLoad(source,p,0);
  let world=textureLoad(positions,p,0);
  let normal=textureLoad(normals,p,0).xyz;
  if(u.flags.w<0.5 || world.w==0.0 || dot(normal,normal)<=1e-6) { return raw; }
  let depth=(u.view*vec4f(world.xyz,1)).z;
  var sum=raw.xy*u.filterParams.x;
  var weight=u.filterParams.x;
  let offsets=array<vec2i,4>(vec2i(-1,0),vec2i(1,0),vec2i(0,-1),vec2i(0,1));
  for(var i=0u;i<4u;i++) {
    let q=p+offsets[i];
    if(any(q<vec2i(0))||any(q>=vec2i(u.params.xy))) { continue; }
    let neighbor=textureLoad(positions,q,0);
    let n=textureLoad(normals,q,0).xyz;
    if(neighbor.w==0.0||dot(n,n)<=1e-6) { continue; }
    if(abs((u.view*vec4f(neighbor.xyz,1)).z-depth)>u.filterParams.z) { continue; }
    if(clamp(dot(normal,n),-1.0,1.0)<u.filterParams.w) { continue; }
    sum+=textureLoad(source,q,0).xy*u.filterParams.y;weight+=u.filterParams.y;
  }
  if(weight<=1.1920929e-7) { return raw; }
  return vec4f(clamp(sum/weight,vec2f(0),vec2f(1)),0,1);
}`;
  const taa = `
fn historySurfaceMatches(q: vec2i,world: vec4f,normal: vec3f) -> bool {
  if(u.temporal.z<0.5) {return true;}
  let old=textureLoad(previousPositions,q,0);
  if(world.w==0.0) {return old.w==0.0;}
  if(old.w!=world.w) {return false;}
  let oldNormal=textureLoad(previousNormals,q,0).xyz;
  return dot(oldNormal,normal)>0.95 && abs(dot(old.xyz-world.xyz,normal))<0.01;
}
fn historyVisible(screen: vec2f,world: vec4f,normal: vec3f) -> bool {
  let p=clamp(screen-0.5,vec2f(0),u.params.xy-1.0);let f=fract(p);
  let lo=vec2i(floor(p));let hi=min(lo+1,vec2i(u.params.xy)-1);
  for(var y=0;y<2;y++) {for(var x=0;x<2;x++) {
    let q=vec2i(select(lo.x,hi.x,x==1),select(lo.y,hi.y,y==1));
    let w=select(1.0-f.x,f.x,x==1)*select(1.0-f.y,f.y,y==1);
    if(w>1e-5 && historySurfaceMatches(q,world,normal)) {return true;}
  }}
  return false;
}
fn bilinearHistory(screen: vec2f,world: vec4f,normal: vec3f) -> vec4f {
  let p=clamp(screen-0.5,vec2f(0),u.params.xy-1.0);
  let lo=vec2i(floor(p));let hi=min(lo+1,vec2i(u.params.xy)-1);let f=fract(p);
  var sum=vec3f(0);var weight=0.0;
  for(var y=0;y<2;y++) {for(var x=0;x<2;x++) {
    let q=vec2i(select(lo.x,hi.x,x==1),select(lo.y,hi.y,y==1));
    let w=select(1.0-f.x,f.x,x==1)*select(1.0-f.y,f.y,y==1);
    if(w>0.0 && historySurfaceMatches(q,world,normal)) {sum+=textureLoad(history,q,0).xyz*w;weight+=w;}
  }}
  if(weight<=1e-5) {return vec4f(0);}
  return vec4f(sum/weight,1);
}
fn process(p: vec2i) -> vec4f {
  let world=textureLoad(positions,p,0);
  let sky=world.w==0.0;
  let current=sourceAt(p);
  var result=current;var previous=current;var motion=0.0;
  var valid=u.jitter.z>0.5 && u.taa.z>0.5;
  if(valid) {
    let normal=textureLoad(normals,p,0).xyz;
    if(sky) {
      let old=bilinearHistory(vec2f(p)+0.5,world,normal);previous=old.xyz;valid=old.w>0.5;
    } else {
      let clip=u.previousVP*vec4f(world.xyz,1);
      valid=clip.w>0.05 && all(abs(clip)<vec4f(3.402823e38));
      if(valid) {
        let projected=(clip.xy/clip.w*0.5+0.5)*u.params.xy;
        let screen=projected-u.jitter.xy;
        valid=all(screen>=vec2f(0))&&all(screen<=u.params.xy);
        if(valid && u.temporal.z>0.5) {valid=historyVisible(projected-u.temporal.xy,world,normal);}
        if(valid) {
          motion=length(screen-(vec2f(p)+0.5));
          let old=bilinearHistory(screen,world,normal);previous=old.xyz;valid=old.w>0.5;
        }
      }
    }
  }
  if(valid && u.taa.y>0.5) {
    var lo=current;var hi=current;
    for(var y=-1;y<=1;y++) { for(var x=-1;x<=1;x++) {
      let neighbor=sourceAt(p+vec2i(x,y));lo=min(lo,neighbor);hi=max(hi,neighbor);
    } }
    previous=clamp(previous,lo,hi);
  }
  let blend=select(u.taa.x,max(u.taa.x,min(0.25,motion*0.1)),u.temporal.z>0.5);
  if(valid) { result=previous+(current-previous)*blend; }
  return vec4f(result,select(0.0,1.0,valid&&!sky));
}`;
  const bloomExtract = `
fn bright(c: vec3f) -> vec3f {
  let l=max(c.x,max(c.y,c.z));let knee=clamp(l-0.75,0.0,1.5);
  return c*(max(l-1.5,knee*knee/3.0)/max(l,1e-5));
}
fn process(p: vec2i) -> vec4f {
  var sum=vec3f(0);
  for(var y=0;y<2;y++) {for(var x=0;x<2;x++) {sum+=bright(sourceAt(p*2+vec2i(x,y)));}}
  return vec4f(sum*0.25,1);
}`;
  const bloomBlur = horizontal => `
fn process(p: vec2i) -> vec4f {
  let dims=vec2i(textureDimensions(source));
  let sigma=clamp(u.params.y/240.0,0.65,4.5);
  var sum=vec3f(0);var weight=0.0;
  for(var i=-14;i<=14;i++) {
    if(abs(f32(i))>ceil(sigma*3.0)) {continue;}
    let w=exp(-0.5*f32(i*i)/(sigma*sigma));
    let q=clamp(p+vec2i(${horizontal?'i,0':'0,i'}),vec2i(0),dims-1);
    sum+=textureLoad(source,q,0).xyz*w;weight+=w;
  }
  return vec4f(sum/weight,1);
}`;
  const present = `
fn bloomAt(pixel: vec2i) -> vec3f {
  let q=(vec2f(pixel)+0.5)*0.5-0.5;
  let f=fract(q);let dims=vec2i(textureDimensions(bloom));
  let lo=clamp(vec2i(floor(q)),vec2i(0),dims-1);
  let hi=clamp(vec2i(floor(q))+1,vec2i(0),dims-1);
  return mix(mix(textureLoad(bloom,lo,0).xyz,textureLoad(bloom,vec2i(hi.x,lo.y),0).xyz,f.x),
    mix(textureLoad(bloom,vec2i(lo.x,hi.y),0).xyz,textureLoad(bloom,hi,0).xyz,f.x),f.y);
}
fn process(p: vec2i) -> vec4f {
  let resolved=textureLoad(source,p,0);
  var result=resolved.xyz;
  if(u.taa.w>0.0 && resolved.w>0.5) {
    let blur=(result*4.0+sourceAt(p+vec2i(-1,0))+sourceAt(p+vec2i(1,0))
              +sourceAt(p+vec2i(0,-1))+sourceAt(p+vec2i(0,1)))/8.0;
    result=max(vec3f(0),result+(result-blur)*u.taa.w);
  }
  if(u.jitter.w>0.5) {result+=bloomAt(p)*0.12;}
  result=max(result*max(u.params.w,0.0),vec3f(0));result=result/(1.0+result);
  result=select(12.92*result,1.055*pow(result,vec3f(1.0/2.4))-0.055,result>=vec3f(0.0031308));
  if(textureLoad(positions,p,0).w!=0.0) {
    let bayer=array<i32,16>(0,8,2,10,12,4,14,6,3,11,1,9,15,7,13,5);
    result+=(f32(bayer[(p.y&3)*4+(p.x&3)])-7.5)*(2.0/16.0)/255.0;
  }
  return vec4f(clamp(result,vec3f(0),vec3f(1)),1);
}`;
  const stages = {filter, taa, present, bloomExtract, bloomX:bloomBlur(true), bloomY:bloomBlur(false)};
  if (!Object.hasOwn(stages, stage)) throw new Error(`Unknown WebGPU post stage: ${stage}`);
  return bindings + stages[stage] + `
@compute @workgroup_size(8,8) fn computeMain(@builtin(global_invocation_id) id: vec3u) {
  if(any(id.xy>=textureDimensions(destination))) { return; }
  let p=vec2i(id.xy);textureStore(destination,p,process(p));
}`;
}

export { glShadowFilter, glTaaResolve, glPostPresent, glBloomExtract, glBloomBlur, webgpuPostShader };
