import { webgpuPostShader } from './gpu-post.js';

// Deferred WebGPU prototype: rasterized G-buffer -> compute lighting -> present.
const webgpuUniforms = `
struct Light { direction: vec4f, color: vec4f }
struct PointLight { position: vec4f, color: vec4f }
struct Uniforms {
  view: mat4x4f,
  camera: vec4f,
  skyTop: vec4f, // w: sparse page-table byte length
  skyHorizon: vec4f, // w: diffuse cache byte offset
  ambientTop: vec4f,
  ground: vec4f, // w: sparse-volume flag
  lights: array<Light, 2>,
  origin: vec4f,
  grid: vec4f,
  params: vec4f,
  flags: vec4f,
  jitter: vec4f,
  albedo: array<vec4f, 16>,
  materials: array<vec4f, 16>,
  previousVP: mat4x4f,
  taa: vec4f,
  filterParams: vec4f,
  pointLights: array<PointLight,6>,
  temporal: vec4f, // previous G-buffer jitter, camera translation flag
}
@group(0) @binding(0) var<uniform> u: Uniforms;
`;
const webgpuGeometry = webgpuUniforms + `
struct VertexOut {
  @builtin(position) position: vec4f,
  @location(0) world: vec3f,
  @location(1) normal: vec3f,
  @location(2) ao: f32,
  @location(3) @interpolate(flat) material: u32,
}
@vertex fn vertexMain(@location(0) position: vec3f, @location(1) normal: vec3f,
                      @location(2) ao: f32, @location(3) material: f32) -> VertexOut {
  let p = (u.view * vec4f(position, 1)).xyz;
  let a = 1000.0 / (1000.0 - 0.05);
  var out: VertexOut;
  out.position = vec4f(p.x * 1.6 * u.params.y / u.params.x, p.y * 1.6 - u.camera.w * 2.0 * p.z,
                      a * p.z - 0.05 * a, p.z);
  out.position.x -= u.jitter.x * 2.0 / u.params.x * p.z;
  out.position.y += u.jitter.y * 2.0 / u.params.y * p.z;
  out.world = position; out.normal = normal; out.ao = ao; out.material = u32(material);
  return out;
}
struct GBuffer {
  @location(0) positionMaterial: vec4f,
  @location(1) normalAO: vec4f,
}
@fragment fn fragmentMain(in: VertexOut) -> GBuffer {
  if (dot(in.normal, u.camera.xyz - in.world) <= 0.0) { discard; }
  var out: GBuffer;
  out.positionMaterial = vec4f(in.world, f32(in.material + 1u));
  out.normalAO = vec4f(in.normal, in.ao);
  return out;
}
`;
const webgpuGeometryCompact = webgpuGeometry.replace(
  /@vertex fn vertexMain\([\s\S]*? -> VertexOut \{/,
  `@vertex fn vertexMain(@builtin(vertex_index) vertex: u32,
    @location(0) anchor: vec3f, @location(1) edgeA: vec3f, @location(2) edgeB: vec3f,
    @location(3) params: vec2f, @location(4) visibility: vec4f) -> VertexOut {
    let corners=array<vec2f,6>(vec2f(0,0),vec2f(1,0),vec2f(1,1),vec2f(0,0),vec2f(1,1),vec2f(0,1));
    let indices=array<u32,6>(0u,1u,2u,0u,2u,3u);
    let normals=array<vec3f,6>(vec3f(0,1,0),vec3f(0,-1,0),vec3f(-1,0,0),vec3f(1,0,0),vec3f(0,0,-1),vec3f(0,0,1));
    let uv=corners[vertex];let position=anchor+edgeA*uv.x+edgeB*uv.y;
    let normal=normals[u32(params.x)];let material=params.y;let ao=visibility[indices[vertex]];`);
const webgpuLighting = webgpuUniforms + `
@group(0) @binding(1) var<storage, read> voxels: array<u32>;
@group(0) @binding(2) var positions: texture_2d<f32>;
@group(0) @binding(3) var normals: texture_2d<f32>;
@group(0) @binding(4) var masks: texture_2d<f32>;
@group(0) @binding(5) var outputFrame: texture_storage_2d<rgba32float, write>;

fn outputColor(input: vec3f) -> vec3f {
  var c = max(input * u.params.w, vec3f(0));
  c = c / (1.0 + c);
  return select(12.92 * c, 1.055 * pow(c, vec3f(1.0 / 2.4)) - 0.055, c >= vec3f(0.0031308));
}
fn randomValue(pixel: vec2f, salt: f32) -> f32 {
  return fract(sin(dot(pixel, vec2f(12.9898, 78.233)) + u.params.z * 1.618 + salt * 37.17) * 43758.5453);
}
fn basis(normal: vec3f) -> mat3x3f {
  let axis = select(vec3f(1, 0, 0), vec3f(0, 1, 0), abs(normal.y) < 0.95);
  let t = normalize(cross(axis, normal));
  return mat3x3f(t, cross(normal, t), normal);
}
fn byteAt(slot: u32) -> u32 { return (voxels[slot>>2u]>>((slot&3u)*8u))&255u; }
fn pageAt(p: vec3i) -> u32 {
  let b=vec3u(p)/8u;let size=vec3u(u.grid.xyz)/8u;
  return voxels[b.x+size.x*(b.y+size.y*b.z)];
}
fn voxelAt(p: vec3i) -> u32 {
  let size=vec3u(u.grid.xyz);let q=vec3u(p);
  if(u.ground.w<0.5) {return byteAt(q.x+size.x*(q.y+size.y*q.z));}
  let page=pageAt(p);if(page==0u) {return 0u;}
  let k=q%8u;
  return byteAt(u32(u.skyTop.w)+(page-1u)*512u+k.x+8u*k.y+64u*k.z);
}
struct Hit { material: i32, position: vec3f, normal: vec3f }
fn trace(start: vec3f, direction: vec3f, limit: f32) -> Hit {
  let miss = Hit(-1, vec3f(0), vec3f(0));
  let g = (start - u.origin.xyz) / u.origin.w;
  let d = direction / u.origin.w;
  let safeD = select(d, vec3f(1e-8), abs(d) < vec3f(1e-8));
  let inv = 1.0 / safeD;
  let t0 = -g * inv;
  let t1 = (u.grid.xyz - g) * inv;
  let lo = min(t0, t1); let hi = max(t0, t1);
  let enter = max(0.0, max(lo.x, max(lo.y, lo.z)));
  let leave = min(limit, min(hi.x, min(hi.y, hi.z)));
  if (enter > leave) { return miss; }
  var cell = vec3i(floor(g + d * (enter + 0.0001)));
  let stepDir = vec3i(sign(safeD));
  let delta = abs(inv);
  var next = (vec3f(cell) + step(vec3f(0), safeD) - g) * inv;
  var distance = enter;
  var normal = -direction;
  let size = vec3i(u.grid.xyz);
  for (var i = 0; i < size.x + size.y + size.z + 3; i++) {
    if (any(cell < vec3i(0)) || any(cell >= size) || distance > leave) { return miss; }
    let value = voxelAt(cell);
    // Jump to the next empty-brick boundary, then resume exact voxel DDA.
    // The crossed axis is set explicitly, avoiding floating-point cracks.
    if(u.ground.w>0.5 && pageAt(cell)==0u) {
      let brick=cell/8;
      let edge=(vec3f(brick)*8.0+step(vec3f(0),safeD)*8.0-g)*inv;
      var axis=2u;
      if(edge.x<edge.y) {if(edge.x<edge.z) {axis=0u;}} else if(edge.y<edge.z) {axis=1u;}
      distance=edge[axis];
      cell=clamp(vec3i(floor(g+d*distance)),brick*8,brick*8+vec3i(7));
      cell[axis]=brick[axis]*8+select(-1,8,stepDir[axis]>0);
      next=(vec3f(cell)+step(vec3f(0),safeD)-g)*inv;
      normal=vec3f(0);normal[axis]=-f32(stepDir[axis]);continue;
    }
    if (value > 0u && value < 128u) { return Hit(i32(value) - 1, start + direction * distance, normal); }
    var axis = 2u;
    if (next.x < next.y) { if (next.x < next.z) { axis = 0u; } }
    else if (next.y < next.z) { axis = 1u; }
    distance = next[axis]; next[axis] += delta[axis]; cell[axis] += stepDir[axis];
    normal = vec3f(0); normal[axis] = -f32(stepDir[axis]);
  }
  return miss;
}
// Primary visibility repair for subpixel T-junction cracks in greedy quads.
// Only empty raster samples bordering geometry are traced; real openings stay
// empty when the ray exits the voxel volume. No wall/roof geometry is expanded.
fn occupiedCell(p: vec3i) -> bool {
  if(any(p<vec3i(0))||any(p>=vec3i(u.grid.xyz))) {return false;}
  let value=voxelAt(p);return value>0u && value<128u;
}
fn hitContactAO(position: vec3f,normal: vec3f) -> f32 {
  let g=(position-u.origin.xyz)/u.origin.w;
  let cell=vec3i(floor(g-normal*0.01));
  let f=clamp(g-vec3f(cell),vec3f(0),vec3f(1));
  let axis=select(select(2u,1u,abs(normal.y)>0.5),0u,abs(normal.x)>0.5);
  let a=(axis+1u)%3u;let b=(axis+2u)%3u;let air=cell+vec3i(round(normal));
  var ao=0.0;
  for(var j=0;j<2;j++) {for(var i=0;i<2;i++) {
    var sideA=air;var sideB=air;var diagonal=air;
    sideA[a]+=select(-1,1,i==1);sideB[b]+=select(-1,1,j==1);
    diagonal[a]=sideA[a];diagonal[b]=sideB[b];
    let sa=occupiedCell(sideA);let sb=occupiedCell(sideB);
    let count=select(i32(sa)+i32(sb)+i32(occupiedCell(diagonal)),3,sa&&sb);
    let weight=select(1.0-f[a],f[a],i==1)*select(1.0-f[b],f[b],j==1);
    ao+=(1.0-0.2*f32(count))*weight;
  }}
  return ao;
}
struct RepairedGBuffer {
  @location(0) positionMaterial: vec4f,
  @location(1) normalAO: vec4f,
}
@vertex fn repairVertex(@builtin(vertex_index) vertex: u32) -> @builtin(position) vec4f {
  let p=vec2f(f32((vertex<<1u)&2u),f32(vertex&2u));return vec4f(p*2.0-1.0,0,1);
}
@fragment fn repairFragment(@builtin(position) screen: vec4f) -> RepairedGBuffer {
  let pixel=vec2i(screen.xy);let size=vec2i(u.params.xy);
  if(textureLoad(positions,pixel,0).w!=0.0) {discard;}
  let offsets=array<vec2i,4>(vec2i(-1,0),vec2i(1,0),vec2i(0,-1),vec2i(0,1));
  var neighbors=0;
  for(var i=0u;i<4u;i++) {
    let q=clamp(pixel+offsets[i],vec2i(0),size-1);
    if(textureLoad(positions,q,0).w!=0.0) {neighbors++;}
  }
  if(neighbors<2) {discard;}
  let samplePosition=screen.xy+u.jitter.xy;
  let viewRay=vec3f((samplePosition.x-u.params.x*0.5)/(u.params.y*0.8),
    -(samplePosition.y-u.params.y*(0.5+u.camera.w))/(u.params.y*0.8),1);
  let direction=normalize(vec3f(dot(u.view[0].xyz,viewRay),dot(u.view[1].xyz,viewRay),dot(u.view[2].xyz,viewRay)));
  let hit=trace(u.camera.xyz+direction*(0.05*length(viewRay)),direction,1000.0);
  if(hit.material<0) {discard;}
  var result: RepairedGBuffer;
  result.positionMaterial=vec4f(hit.position,f32(hit.material+1));
  result.normalAO=vec4f(hit.normal,hitContactAO(hit.position,hit.normal));
  return result;
}
fn cachedAmbient(position: vec3f, normal: vec3f) -> f32 {
  if(u.jitter.w<0.5) { return 1.0; }
  let p=vec3i(floor((position+normal*0.04-u.origin.xyz)/u.origin.w));
  let size=vec3i(u.grid.xyz);
  if(any(p<vec3i(0))||any(p>=size)) {return 1.0;}
  var value=voxelAt(p);
  if(value>0u && value<128u) {return 0.0;}
  if(value==0u && u.ground.w>0.5) {
    let c=vec3u(p)/4u;let dims=vec3u(size)/4u;
    value=byteAt(u32(u.skyHorizon.w)+c.x+dims.x*(c.y+dims.y*c.z));
  }
  return f32(value&127u)/127.0*select(1.0,0.32,normal.y<0.0);
}
fn skyIncoming(normal: vec3f) -> vec3f {
  return mix(u.ground.xyz, u.ambientTop.xyz, normal.y * 0.5 + 0.5) * u.grid.w;
}
fn shade(world: vec3f, normal: vec3f, ao: f32, material: u32, pixel: vec2f, mask: vec2f) -> vec3f {
  let visibility = select(cachedAmbient(world,normal), ao, u.flags.x > 0.5) * select(1.0,cachedAmbient(world,normal),u.ground.w>0.5 && u.flags.x>0.5);
  var base = u.albedo[material].xyz;
  if (u.jitter.w > 0.5) {
    let cell=floor((world-normal*0.002)*24.0);
    let grain=fract(sin(dot(cell,vec3f(12.9898,78.233,37.719)))*43758.5453);
    base*=0.86+0.22*grain;
    if(material==3u) { base*=0.80+0.25*sin(world.x*93.0+sin(world.z*7.0)); }
  }
  let params = u.materials[material].xyz;
  var result = base * (skyIncoming(normal) * visibility + u.materials[material].w);
  for(var i=0;i<6;i++) {
    let light=u.pointLights[i];let delta=light.position.xyz-world;
    let d2=dot(delta,delta);let r2=light.color.w*light.color.w;
    if(light.position.w<=0.0 || d2>=r2 || d2<1e-8) { continue; }
    let distance=sqrt(d2);let direction=delta/distance;
    let cosine=max(dot(normal,direction),0.0);
    if(cosine<=0.0) { continue; }
    if(u.flags.y>0.5 && trace(world+normal*0.04,direction,max(0.0,distance-0.08)).material>=0) { continue; }
    let fade=1.0-d2/r2;
    result+=base*light.color.xyz*(cosine*light.position.w*fade*fade/(1.0+d2*1.8));
  }
  let view = normalize(u.camera.xyz - world);
  for (var i = 0u; i < 2u; i++) {
    let light = u.lights[i];
    let l = light.direction.xyz;
    let ndl = max(dot(normal, l), 0.0);
    let intensity = light.direction.w;
    if (ndl <= 0.0 || intensity <= 0.0) { continue; }
    let visible = mask[i];
    let h = normalize(l + view);
    let fresnel = params.y + (1.0 - params.y) * pow(1.0 - max(dot(view, h), 0.0), 5.0);
    let spec = (params.z + 8.0) / (8.0 * 3.14159265) * pow(max(dot(normal, h), 0.0), params.z) * fresnel * ndl;
    result += (base * params.x * (1.0 - fresnel) * ndl * intensity
               + vec3f(clamp(spec * intensity, 0.0, 1.0))) * light.color.xyz * visible;
  }
  if (u.flags.z > 0.5) {
    var p = world; var norm = normal; var throughput = base; var indirect = vec3f(0);
    for (var bounce = 0u; bounce < 2u; bounce++) {
      let sample = randomValue(pixel, 10.0 + f32(bounce) * 2.0);
      let angle = randomValue(pixel, 11.0 + f32(bounce) * 2.0) * 6.283185;
      let cosine = sqrt(1.0 - sample);
      let ray = basis(norm) * vec3f(sqrt(sample) * cos(angle), sqrt(sample) * sin(angle), cosine);
      let hit = trace(p + norm * 0.04, ray, select(12.0,1000.0,u.jitter.w>0.5));
      if (hit.material < 0) {
        if(u.jitter.w>0.5) { indirect+=throughput*skyIncoming(ray)*cosine; }
        break;
      }
      var incoming = skyIncoming(hit.normal);
      let hitOrigin=hit.position+hit.normal*0.04;
      incoming*=cachedAmbient(hit.position,hit.normal);
      incoming+=vec3f(u.materials[u32(hit.material)].w);
      for (var i = 0u; i < 2u; i++) {
        let l=u.lights[i];let nl=max(dot(hit.normal,l.direction.xyz),0.0);
        if(nl>0.0 && l.direction.w>0.0 && trace(hitOrigin,l.direction.xyz,1000.0).material<0) {
          incoming+=l.color.xyz*l.direction.w*nl;
        }
      }
      throughput *= u.albedo[u32(hit.material)].xyz * cosine;
      indirect += incoming * throughput;
      p = hit.position; norm = hit.normal;
    }
    result += clamp(indirect, vec3f(0), vec3f(4)) * min(1.0, visibility + 0.15);
  }
  // One rough specular ray from polished floor tiles, through the actual voxel scene.
  if(u.jitter.w>0.5 && (material==4u || material==11u) && normal.y>0.9) {
    let a=randomValue(pixel,51.0)*6.283185;
    let r=sqrt(randomValue(pixel,52.0))*0.07;
    let reflection=reflect(-view,normal);
    var ray=normalize(reflection+basis(reflection)*vec3f(cos(a)*r,sin(a)*r,0));
    ray=normalize(ray+normal*max(0.0,0.002-dot(ray,normal)));
    let hit=trace(world+normal*0.04,ray,1000.0);
    var reflected=u.skyHorizon.xyz;
    if(hit.material>=0) {
      let m=u32(hit.material);
      reflected=u.albedo[m].xyz*(skyIncoming(hit.normal)*cachedAmbient(hit.position,hit.normal)+u.materials[m].w);
      for(var i=0u;i<2u;i++) {
        let light=u.lights[i];let ndl=max(dot(hit.normal,light.direction.xyz),0.0);
        if(ndl>0.0 && light.direction.w>0.0) {
          let blocker=trace(hit.position+hit.normal*0.05,light.direction.xyz,1000.0);
          if(blocker.material<0) { reflected+=u.albedo[m].xyz*light.color.xyz*light.direction.w*ndl; }
        }
      }
    }
    let fresnel=0.10+0.45*pow(1.0-max(dot(normal,view),0.0),5.0);
    result=mix(result,reflected,fresnel);
  }
  return result;
}
@compute @workgroup_size(8,8) fn shadowMain(@builtin(global_invocation_id) id: vec3u) {
  let size=vec2u(u.params.xy);
  if(any(id.xy>=size)) { return; }
  let pixel=vec2i(id.xy);
  let world=textureLoad(positions,pixel,0);
  let normal=textureLoad(normals,pixel,0).xyz;
  let samplePixel=vec2f(f32(id.x)+0.5,f32(size.y-id.y)-0.5);
  var mask=vec2f(1);
  if(u.flags.y>0.5 && world.w!=0.0) {
    for(var i=0u;i<2u;i++) {
      let light=u.lights[i];let l=light.direction.xyz;
      if(light.direction.w<=0.0 || dot(normal,l)<=0.0) { continue; }
      let angle=randomValue(samplePixel,f32(i)*2.0)*6.283185;
      let radius=sqrt(randomValue(samplePixel,f32(i)*2.0+1.0))*tan(light.color.w);
      let ray=normalize(l+basis(l)*vec3f(cos(angle)*radius,sin(angle)*radius,0));
      if(dot(normal,ray)>0.0) {
        let hit=trace(world.xyz+normal*0.05,ray,1000.0);
        mask[i]=select(0.0,1.0,hit.material<0);
      }
    }
  }
  textureStore(outputFrame,pixel,vec4f(mask,0,1));
}
@compute @workgroup_size(8,8) fn lightingMain(@builtin(global_invocation_id) id: vec3u) {
  let size=vec2u(u.params.xy);
  if(any(id.xy>=size)) { return; }
  let pixel=vec2i(id.xy);
  let position=textureLoad(positions,pixel,0);
  var linear: vec3f;
  if(position.w<0.5) {
    var t=0.0;
    if(size.y>1u) { t=f32(id.y)/f32(size.y-1u); }
    linear=mix(u.skyTop.xyz,u.skyHorizon.xyz,t);
  } else {
    let normal=textureLoad(normals,pixel,0);
    linear=shade(position.xyz,normal.xyz,normal.w,u32(position.w)-1u,
                 vec2f(f32(id.x)+0.5,f32(size.y-id.y)-0.5),textureLoad(masks,pixel,0).xy);
  }
  textureStore(outputFrame,pixel,vec4f(linear,1));
}
`;
const webgpuPresent = `
@group(0) @binding(0) var frame: texture_2d<f32>;
@vertex fn vertexMain(@builtin(vertex_index) index: u32) -> @builtin(position) vec4f {
  let p = vec2f(f32((index << 1u) & 2u), f32(index & 2u));
  return vec4f(p * 2.0 - 1.0, 0, 1);
}
@fragment fn fragmentMain(@builtin(position) position: vec4f) -> @location(0) vec4f {
  return textureLoad(frame, vec2i(position.xy), 0);
}
`;

export class WebGPURenderer {
  static async create(canvas, scene) {
    if (!navigator.gpu) throw new Error('浏览器不支持 WebGPU，或页面不在安全上下文中');
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error('未找到可用的 WebGPU adapter');
    const timestamp = adapter.features.has('timestamp-query');
    const device = await adapter.requestDevice({requiredFeatures: timestamp ? ['timestamp-query'] : []});
    const renderer = new WebGPURenderer(canvas, device, scene, timestamp);
    renderer.adapterInfo = adapter.info;
    try {
      await renderer.initialize();
      return renderer;
    } catch (error) {
      device.destroy();
      throw error;
    }
  }

  constructor(canvas, device, scene, timestamp) {
    this.canvas = canvas; this.device = device; this.scene = scene;
    this.frame = 0; this.valid = false; this.gpuMs = null; this.generation = 0;
    this.targets = []; this.timingPending = false; this.error = null;
    this.timestamp = timestamp;
    device.lost.then(info => { this.error = `WebGPU 设备丢失：${info.message || info.reason}`; });
    device.addEventListener('uncapturederror', event => { this.error = event.error.message; });
  }

  buffer(data, usage) {
    const buffer = this.device.createBuffer({size: data.byteLength, usage, mappedAtCreation: true});
    new Uint8Array(buffer.getMappedRange()).set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength));
    buffer.unmap();
    return buffer;
  }

  async initialize() {
    const device = this.device;
    this.context = this.canvas.getContext('webgpu');
    if (!this.context) throw new Error('无法创建 WebGPU canvas');
    this.format = navigator.gpu.getPreferredCanvasFormat();
    this.context.configure({device, format: this.format, alphaMode: 'opaque'});
    device.pushErrorScope('validation');
    const compact=Boolean(this.scene.info[176]);
    const postStages = ['filter', 'taa', 'present', 'bloomExtract', 'bloomX', 'bloomY'];
    const shaders = [
      {label: compact ? 'geometry-compact' : 'geometry', code: compact ? webgpuGeometryCompact : webgpuGeometry},
      {label: 'lighting', code: webgpuLighting},
      {label: 'canvas-present', code: webgpuPresent},
      ...postStages.map(stage => ({label: `post-${stage}`, code: webgpuPostShader(webgpuUniforms, stage)})),
    ];
    const modules = shaders.map(shader => device.createShaderModule(shader));
    for (const module of modules) {
      const info = await module.getCompilationInfo();
      const errors = info.messages.filter(message => message.type === 'error');
      if (errors.length) {
        await device.popErrorScope();
        throw new Error(errors.map(error => `${module.label} ${error.lineNum}:${error.linePos} ${error.message}`).join('\n'));
      }
    }
    this.geometry = await device.createRenderPipelineAsync({
      layout: 'auto',
      vertex: {module: modules[0], entryPoint: 'vertexMain', buffers: compact?[{arrayStride:60,stepMode:'instance',attributes:[
        {shaderLocation:0,offset:0,format:'float32x3'},{shaderLocation:1,offset:12,format:'float32x3'},
        {shaderLocation:2,offset:24,format:'float32x3'},{shaderLocation:3,offset:36,format:'float32x2'},
        {shaderLocation:4,offset:44,format:'float32x4'}]}]:[{arrayStride: 32, attributes: [
        {shaderLocation: 0, offset: 0, format: 'float32x3'},
        {shaderLocation: 1, offset: 12, format: 'float32x3'},
        {shaderLocation: 2, offset: 24, format: 'float32'},
        {shaderLocation: 3, offset: 28, format: 'float32'},
      ]}]},
      fragment: {module: modules[0], entryPoint: 'fragmentMain', targets: [{format: 'rgba32float'}, {format: 'rgba16float'}]},
      primitive: {topology: 'triangle-list'},
      depthStencil: {format: 'depth24plus', depthWriteEnabled: true, depthCompare: 'less'},
    });
    this.repair = await device.createRenderPipelineAsync({
      label:'voxel-primary-visibility',layout:'auto',
      vertex:{module:modules[1],entryPoint:'repairVertex'},
      fragment:{module:modules[1],entryPoint:'repairFragment',targets:[{format:'rgba32float'},{format:'rgba16float'}]},
      primitive:{topology:'triangle-list'},
    });
    this.lighting = await device.createComputePipelineAsync({layout: 'auto', compute: {module: modules[1], entryPoint: 'lightingMain'}});
    this.shadow = await device.createComputePipelineAsync({layout: 'auto', compute: {module: modules[1], entryPoint: 'shadowMain'}});
    this.post = {};
    for (const [index, stage] of postStages.entries()) {
      this.post[stage] = await device.createComputePipelineAsync({layout: 'auto', compute: {module: modules[index + 3], entryPoint: 'computeMain'}});
    }
    this.present = await device.createRenderPipelineAsync({
      layout: 'auto', vertex: {module: modules[2], entryPoint: 'vertexMain'},
      fragment: {module: modules[2], entryPoint: 'fragmentMain', targets: [{format: this.format}]},
    });
    this.uniform = device.createBuffer({size: 276 * 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST});
    this.mesh = this.buffer(this.scene.mesh, GPUBufferUsage.VERTEX);
    // Four byte-sized palette indices per u32; avoids a 4x storage expansion.
    const packed = new Uint8Array(Math.ceil(this.scene.voxels.length / 4) * 4);
    packed.set(this.scene.voxels);
    this.voxels = this.buffer(new Uint32Array(packed.buffer), GPUBufferUsage.STORAGE);
    this.geometryGroup = device.createBindGroup({layout: this.geometry.getBindGroupLayout(0), entries: [{binding: 0, resource: {buffer: this.uniform}}]});
    if (this.timestamp) {
      this.query = device.createQuerySet({type: 'timestamp', count: 2});
      this.resolveBuffer = device.createBuffer({size: 16, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC});
      this.readback = device.createBuffer({size: 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ});
    }
    const error = await device.popErrorScope();
    if (error) throw new Error(error.message);
  }

  reset() {
    this.valid = false; this.gpuMs = null; this.generation++;
  }

  dispose() {
    this.reset();
    this.context?.unconfigure();
    this.device.destroy();
  }

  resize(width, height) {
    if (this.targets.length && this.canvas.width === width && this.canvas.height === height) return;
    for (const texture of this.targets) texture.destroy();
    this.canvas.width = width; this.canvas.height = height; this.reset();
    const texture = (format, usage, w=width, h=height) => this.device.createTexture({size: [w, h], format, usage});
    const transfers = GPUTextureUsage.COPY_SRC | GPUTextureUsage.COPY_DST;
    const attachment = GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING | transfers;
    this.position = texture('rgba32float', attachment);
    this.normal = texture('rgba16float', attachment);
    this.rasterCoverage=texture('rgba32float',attachment,this.scene.info[62]?width:1,this.scene.info[62]?height:1);
    this.previousPosition=texture('rgba32float',attachment);
    this.previousNormal=texture('rgba16float',attachment);
    this.depth = texture('depth24plus', GPUTextureUsage.RENDER_ATTACHMENT);
    const outputUsage = GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | transfers;
    this.rawShadow = texture('rgba32float', outputUsage);
    this.filteredShadow = texture('rgba32float', outputUsage);
    this.linear = texture('rgba32float', outputUsage);
    this.history = [texture('rgba32float', outputUsage), texture('rgba32float', outputUsage)];
    this.output = texture('rgba8unorm', outputUsage);
    this.bloom=[texture('rgba32float',outputUsage,Math.ceil(width/2),Math.ceil(height/2)),texture('rgba32float',outputUsage,Math.ceil(width/2),Math.ceil(height/2))];
    this.targets = [this.position, this.normal, this.rasterCoverage, this.previousPosition, this.previousNormal, this.depth, this.rawShadow, this.filteredShadow, this.linear, ...this.history, this.output, ...this.bloom];
    const uniform = {buffer: this.uniform}, voxels = {buffer: this.voxels};
    const pos = this.position.createView(), normal = this.normal.createView();
    this.repairGroup=this.group(this.repair,{0:uniform,1:voxels,2:this.rasterCoverage.createView()});
    this.shadowGroup = this.group(this.shadow, {0: uniform, 1: voxels, 2: pos, 3: normal, 5: this.rawShadow.createView()});
    this.filterGroup = this.group(this.post.filter, {0: uniform, 1: pos, 2: normal, 3: this.rawShadow.createView(), 5: this.filteredShadow.createView()});
    this.lightingGroup = this.group(this.lighting, {0: uniform, 1: voxels, 2: pos, 3: normal, 4: this.filteredShadow.createView(), 5: this.linear.createView()});
    this.taaGroups = this.history.map((output, index) => this.group(this.post.taa, {
      0: uniform, 1: pos, 2:normal, 3: this.linear.createView(), 4: this.history[1-index].createView(), 5: output.createView(),
      7:this.previousPosition.createView(),8:this.previousNormal.createView(),
    }));
    this.outputGroups = this.history.map(source => this.group(this.post.present, {
      0: uniform, 1: pos, 3: source.createView(), 5: this.output.createView(), 6:this.bloom[0].createView(),
    }));
    this.bloomExtractGroups=this.history.map(source=>this.group(this.post.bloomExtract,{
      0:uniform,3:source.createView(),5:this.bloom[0].createView()}));
    this.bloomXGroup=this.group(this.post.bloomX,{0:uniform,3:this.bloom[0].createView(),5:this.bloom[1].createView()});
    this.bloomYGroup=this.group(this.post.bloomY,{0:uniform,3:this.bloom[1].createView(),5:this.bloom[0].createView()});
    this.presentGroup = this.group(this.present, {0: this.output.createView()});
  }

  group(pipeline, resources) {
    return this.device.createBindGroup({layout: pipeline.getBindGroupLayout(0),
      entries: Object.entries(resources).map(([binding, resource]) => ({binding: Number(binding), resource}))});
  }

  dispatch(encoder, pipeline, group, width, height) {
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline); pass.setBindGroup(0, group);
    pass.dispatchWorkgroups(Math.ceil(width / 8), Math.ceil(height / 8)); pass.end();
  }

  uniforms(data, settings) {
    const s = data.state, info = this.scene.info;
    const values = [...s.subarray(0, 16)];
    const vector = (array, w = 0) => values.push(...array, w);
    vector(s.subarray(16, 19),s[77] || 0); vector(s.subarray(19, 22),info[177]||0); vector(s.subarray(22, 25),info[178]||0);
    vector(s.subarray(25, 28)); vector(s.subarray(28, 31),info[176]||0);
    values.push(...s.subarray(33, 49));
    vector(info.subarray(4, 7), info[7]);
    values.push(info[2], info[3], info[2], s[31]);
    values.push(data.width, data.height, s[76], s[32]);
    values.push(Number(settings.ao), Number(settings.shadows), Number(settings.gi), s[75]);
    values.push(s[65], s[66], s[73], info[62] || 0);
    for (let i = 0; i < 16; i++) {
      const o=i<8?8+i*6:info.length>=127?63+(i-8)*6:8;
      vector(info.subarray(o,o+3));
    }
    for (let i = 0; i < 16; i++) {
      const o=i<8?11+i*6:info.length>=127?66+(i-8)*6:11;
      vector(info.subarray(o,o+3),info[111+i] || 0);
    }
    values.push(...s.subarray(49,65));
    values.push(s[68], s[74], this.valid ? 1 : 0, s[67]);
    values.push(...s.subarray(69,73));
    for(let i=0;i<6;i++) {
      if(i<(info[127]||0)) values.push(...info.subarray(128+i*8,136+i*8));
      else values.push(0,0,0,0,0,0,0,0);
    }
    const translated=data.cameraTranslated ?? (this.lastState?Math.hypot(s[16]-this.lastState[16],s[17]-this.lastState[17],s[18]-this.lastState[18])>1e-5:false);
    values.push(...(data.previousJitter || this.lastState?.subarray(65,67) || [0,0]),Number(translated),0);
    return new Float32Array(values);
  }

  draw(data, settings) {
    if (this.error) throw new Error(this.error);
    const start = performance.now(), device = this.device;
    this.resize(data.width, data.height);
    device.queue.writeBuffer(this.uniform, 0, this.uniforms(data, settings));
    const encoder = device.createCommandEncoder();
    const timed = this.timestamp && !this.timingPending;
    const geometry = encoder.beginRenderPass({
      colorAttachments: [this.position, this.normal].map(texture => ({view: texture.createView(), loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 0]})),
      depthStencilAttachment: {view: this.depth.createView(), depthLoadOp: 'clear', depthStoreOp: 'discard', depthClearValue: 1},
      ...(timed ? {timestampWrites: {querySet: this.query, beginningOfPassWriteIndex: 0}} : {}),
    });
    geometry.setPipeline(this.geometry); geometry.setBindGroup(0, this.geometryGroup);
    geometry.setVertexBuffer(0, this.mesh); if(this.scene.info[176]) geometry.draw(6,this.scene.mesh.length/15); else geometry.draw(this.scene.mesh.length / 8); geometry.end();
    if(this.scene.info[62]) {
      encoder.copyTextureToTexture({texture:this.position},{texture:this.rasterCoverage},[data.width,data.height]);
      const repair=encoder.beginRenderPass({colorAttachments:[this.position,this.normal].map(texture=>({
        view:texture.createView(),loadOp:'load',storeOp:'store'}))});
      repair.setPipeline(this.repair);repair.setBindGroup(0,this.repairGroup);repair.draw(3);repair.end();
    }
    const index = this.frame % 2;
    this.dispatch(encoder, this.shadow, this.shadowGroup, data.width, data.height);
    this.dispatch(encoder, this.post.filter, this.filterGroup, data.width, data.height);
    this.dispatch(encoder, this.lighting, this.lightingGroup, data.width, data.height);
    this.dispatch(encoder, this.post.taa, this.taaGroups[index], data.width, data.height);
    if(this.scene.info[62]) {
      const w=Math.ceil(data.width/2),h=Math.ceil(data.height/2);
      this.dispatch(encoder,this.post.bloomExtract,this.bloomExtractGroups[index],w,h);
      this.dispatch(encoder,this.post.bloomX,this.bloomXGroup,w,h);
      this.dispatch(encoder,this.post.bloomY,this.bloomYGroup,w,h);
    }
    this.dispatch(encoder, this.post.present, this.outputGroups[index], data.width, data.height);
    encoder.copyTextureToTexture({texture:this.position},{texture:this.previousPosition},[data.width,data.height]);
    encoder.copyTextureToTexture({texture:this.normal},{texture:this.previousNormal},[data.width,data.height]);
    const present = encoder.beginRenderPass({
      colorAttachments: [{view: this.context.getCurrentTexture().createView(), loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 1]}],
      ...(timed ? {timestampWrites: {querySet: this.query, endOfPassWriteIndex: 1}} : {}),
    });
    present.setPipeline(this.present); present.setBindGroup(0, this.presentGroup); present.draw(3); present.end();
    if (timed) {
      encoder.resolveQuerySet(this.query, 0, 2, this.resolveBuffer, 0);
      encoder.copyBufferToBuffer(this.resolveBuffer, 0, this.readback, 0, 16);
    }
    device.queue.submit([encoder.finish()]);
    // Bound latency when the detailed cathedral takes longer than one display frame.
    this.pendingFrames = (this.pendingFrames || 0) + 1;
    device.queue.onSubmittedWorkDone().catch(error => {
      this.error ||= error.message;
    }).finally(() => { this.pendingFrames--; });
    this.lastOutput = this.output; this.lastHistory = this.history[index]; this.lastState = data.state.slice();
    this.valid = Boolean(data.state[73]); this.frame++;
    if (timed) {
      this.timingPending = true;
      const generation = this.generation;
      this.readback.mapAsync(GPUMapMode.READ).then(() => {
        const timestamps = new BigUint64Array(this.readback.getMappedRange());
        if (generation === this.generation && timestamps[1] >= timestamps[0]) {
          this.gpuMs = Number(timestamps[1] - timestamps[0]) / 1e6;
          this.gpuSampleVersion = (this.gpuSampleVersion ?? 0) + 1;
        }
        this.readback.unmap();
      }).catch(() => { this.gpuMs = null; }).finally(() => { this.timingPending = false; });
    }
    return {submitMs: performance.now() - start, gpuMs: this.gpuMs};
  }
}
