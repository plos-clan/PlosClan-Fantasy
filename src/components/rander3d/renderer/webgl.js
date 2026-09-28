import { glShadowFilter, glTaaResolve, glPostPresent, glBloomExtract, glBloomBlur } from './gpu-post.js';

const fullVertex = `#version 300 es
precision highp float;
out vec2 uv;
void main() {
  vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
  uv = p; gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}`;
const meshVertex = `#version 300 es
precision highp float;
layout(location=0) in vec3 position;
layout(location=1) in vec3 normal;
layout(location=2) in float visibility;
layout(location=3) in float material;
uniform mat4 view;
uniform vec2 size, jitter;
uniform float verticalShift;
out vec3 world, n;
out float ao;
flat out int mat;
void main() {
  world=position; n=normal; ao=visibility; mat=int(material);
  vec3 p=(view*vec4(position,1)).xyz;
  float a=1000.0/(1000.0-0.05), b=-0.05*a;
  gl_Position=vec4(p.x*1.6*size.y/size.x, p.y*1.6-verticalShift*2.0*p.z, (2.0*a-1.0)*p.z+2.0*b, p.z);
  gl_Position.xy += vec2(-jitter.x,jitter.y)*2.0/size*p.z;
}`;
const compactMeshVertex=meshVertex.replace(
  /layout\(location=0\)[\s\S]*?uniform mat4 view;/,
  `layout(location=0) in vec3 anchor;
   layout(location=1) in vec3 edgeA;
   layout(location=2) in vec3 edgeB;
   layout(location=3) in vec2 params;
   layout(location=4) in vec4 visibility4;
   uniform mat4 view;`).replace('void main() {', `void main() {
    const vec2 corners[6]=vec2[6](vec2(0,0),vec2(1,0),vec2(1,1),vec2(0,0),vec2(1,1),vec2(0,1));
    const int indices[6]=int[6](0,1,2,0,2,3);
    const vec3 normals[6]=vec3[6](vec3(0,1,0),vec3(0,-1,0),vec3(-1,0,0),vec3(1,0,0),vec3(0,0,-1),vec3(0,0,1));
    vec2 uv=corners[gl_VertexID];vec3 position=anchor+edgeA*uv.x+edgeB*uv.y;
    vec3 normal=normals[int(params.x)];float material=params.y,visibility=visibility4[indices[gl_VertexID]];`);
const glLightingBindings = `#version 300 es
precision highp float;
precision highp usampler3D;
uniform usampler3D voxels;
precision highp usampler2D;
uniform usampler2D sparseVoxels;
uniform vec3 sparseInfo;
uniform int sparseTextureWidth;
uint wordAt(uint slot) {return texelFetch(sparseVoxels,ivec2(slot%uint(sparseTextureWidth),slot/uint(sparseTextureWidth)),0).r;}
uint byteAt(uint slot) {return (wordAt(slot>>2u)>>((slot&3u)*8u))&255u;}

uniform sampler2D positionTex, normalTex, shadowTex;
uniform mat4 view;
uniform vec2 size;
uniform ivec3 gridSize;
uniform vec3 origin, camera, ambientTop, ground, skyTop, skyHorizon;
uniform float blockSize, skyScale, frame;
uniform vec4 lightDir[2], lightColor[2];
uniform vec3 albedo[16];
uniform vec4 materialParams[16];
uniform bool cathedral;
uniform vec4 pointPosition[6], pointColor[6];
uniform bool useAO, useShadow, useGI;
out vec4 color;
float randomValue(vec2 p,float salt) {
  return fract(sin(dot(p,vec2(12.9898,78.233))+frame*1.618+salt*37.17)*43758.5453);
}
mat3 basis(vec3 normal) {
  vec3 t=normalize(cross(abs(normal.y)<0.95?vec3(0,1,0):vec3(1,0,0),normal));
  return mat3(t,cross(normal,t),normal);
}
uint pageAt(ivec3 p) {
  uvec3 b=uvec3(p)/8u,size=uvec3(gridSize)/8u;
  return wordAt(b.x+size.x*(b.y+size.y*b.z));
}
uint voxelAt(ivec3 p) {
  if(sparseInfo.x<0.5) return texelFetch(voxels,p,0).r;
  uint page=pageAt(p);if(page==0u) return 0u;
  uvec3 k=uvec3(p)%8u;
  return byteAt(uint(sparseInfo.y)+(page-1u)*512u+k.x+8u*k.y+64u*k.z);
}

int trace(vec3 start,vec3 direction,float limit,out vec3 hit,out vec3 normal) {
  vec3 g=(start-origin)/blockSize, d=direction/blockSize;
  vec3 safeD=mix(d,vec3(1e-8),lessThan(abs(d),vec3(1e-8)));
  vec3 inv=1.0/safeD;
  vec3 t0=-g*inv,t1=(vec3(gridSize)-g)*inv;
  vec3 lo=min(t0,t1),hi=max(t0,t1);
  float enter=max(0.0,max(lo.x,max(lo.y,lo.z)));
  float leave=min(limit,min(hi.x,min(hi.y,hi.z)));
  if(enter>leave) return -1;
  vec3 p=g+d*(enter+0.0001);
  ivec3 cell=ivec3(floor(p));
  ivec3 stepDir=ivec3(sign(safeD));
  vec3 delta=abs(inv);
  vec3 boundary=vec3(cell)+step(vec3(0),safeD);
  vec3 next=(boundary-g)*inv;
  float distance=enter;
  normal=-direction;
  for(int i=0;i<gridSize.x+gridSize.y+gridSize.z+3;i++) {
    if(any(lessThan(cell,ivec3(0)))||any(greaterThanEqual(cell,gridSize))||distance>leave) return -1;
    uint value=voxelAt(cell);
    if(sparseInfo.x>0.5 && pageAt(cell)==0u) {
      ivec3 brick=cell/8;
      vec3 edge=(vec3(brick)*8.0+step(vec3(0),safeD)*8.0-g)*inv;
      int axis=edge.x<edge.y?(edge.x<edge.z?0:2):(edge.y<edge.z?1:2);
      distance=edge[axis];cell=clamp(ivec3(floor(g+d*distance)),brick*8,brick*8+ivec3(7));
      cell[axis]=brick[axis]*8+(stepDir[axis]>0?8:-1);
      next=(vec3(cell)+step(vec3(0),safeD)-g)*inv;
      normal=vec3(0);normal[axis]=-float(stepDir[axis]);continue;
    }
    if(value>0u && value<128u) { hit=start+direction*distance; return int(value)-1; }
    int axis=next.x<next.y?(next.x<next.z?0:2):(next.y<next.z?1:2);
    distance=next[axis]; next[axis]+=delta[axis]; cell[axis]+=stepDir[axis];
    normal=vec3(0); normal[axis]=-float(stepDir[axis]);
  }
  return -1;
}
float cachedAmbient(vec3 position,vec3 normal) {
  if(!cathedral) return 1.0;
  ivec3 p=ivec3(floor((position+normal*0.04-origin)/blockSize));
  if(any(lessThan(p,ivec3(0)))||any(greaterThanEqual(p,gridSize))) return 1.0;
  uint value=voxelAt(p);
  if(value>0u && value<128u) return 0.0;
  if(value==0u && sparseInfo.x>0.5) {
    uvec3 c=uvec3(p)/4u,dims=uvec3(gridSize)/4u;
    value=byteAt(uint(sparseInfo.z)+c.x+dims.x*(c.y+dims.y*c.z));
  }
  return float(value&127u)/127.0*(normal.y<0.0?0.32:1.0);
}
vec3 skyIncoming(vec3 normal) {return mix(ground,ambientTop,normal.y*0.5+0.5)*skyScale;}
`;
const glRepairFragment = glLightingBindings.replace('out vec4 color;', `
layout(location=0) out vec4 repairedPosition;
layout(location=1) out vec4 repairedNormal;
uniform sampler2D coverageTex;
uniform vec2 jitter;
uniform float verticalShift;`) + `
bool occupiedCell(ivec3 p) {
  if(any(lessThan(p,ivec3(0)))||any(greaterThanEqual(p,gridSize))) return false;
  uint value=voxelAt(p);return value>0u && value<128u;
}
float hitContactAO(vec3 position,vec3 normal) {
  vec3 g=(position-origin)/blockSize;ivec3 cell=ivec3(floor(g-normal*0.01));
  vec3 f=clamp(g-vec3(cell),0.0,1.0);
  int axis=abs(normal.x)>0.5?0:abs(normal.y)>0.5?1:2,a=(axis+1)%3,b=(axis+2)%3;
  ivec3 air=cell+ivec3(round(normal));float ao=0.0;
  for(int j=0;j<2;j++) for(int i=0;i<2;i++) {
    ivec3 sideA=air,sideB=air,diagonal=air;
    sideA[a]+=i==1?1:-1;sideB[b]+=j==1?1:-1;diagonal[a]=sideA[a];diagonal[b]=sideB[b];
    bool sa=occupiedCell(sideA),sb=occupiedCell(sideB);
    int count=sa&&sb?3:int(sa)+int(sb)+int(occupiedCell(diagonal));
    float weight=(i==1?f[a]:1.0-f[a])*(j==1?f[b]:1.0-f[b]);ao+=(1.0-0.2*float(count))*weight;
  }
  return ao;
}
void main() {
  ivec2 pixel=ivec2(gl_FragCoord.xy);
  if(texelFetch(coverageTex,pixel,0).w!=0.0) discard;
  const ivec2 offsets[4]=ivec2[4](ivec2(-1,0),ivec2(1,0),ivec2(0,-1),ivec2(0,1));
  int neighbors=0;
  for(int i=0;i<4;i++) if(texelFetch(coverageTex,clamp(pixel+offsets[i],ivec2(0),ivec2(size)-1),0).w!=0.0) neighbors++;
  if(neighbors<2) discard;
  vec2 samplePosition=vec2(gl_FragCoord.x,size.y-gl_FragCoord.y)+jitter;
  vec3 viewRay=vec3((samplePosition.x-size.x*0.5)/(size.y*0.8),
    -(samplePosition.y-size.y*(0.5+verticalShift))/(size.y*0.8),1);
  vec3 direction=normalize(transpose(mat3(view))*viewRay),hit,normal;
  int material=trace(camera+direction*(0.05*length(viewRay)),direction,1000.0,hit,normal);
  if(material<0) discard;
  repairedPosition=vec4(hit,float(material+1));repairedNormal=vec4(normal,hitContactAO(hit,normal));
}`;
const glGeometryFragment = `#version 300 es
precision highp float;
in vec3 world, n;
in float ao;
flat in int mat;
uniform vec3 camera;
layout(location=0) out vec4 positionMaterial;
layout(location=1) out vec4 normalAO;
void main() {
  if(dot(n,camera-world)<=0.0) discard;
  positionMaterial=vec4(world,float(mat+1));normalAO=vec4(n,ao);
}`;
const glRawShadow = glLightingBindings + `
void main() {
  ivec2 pixel=ivec2(gl_FragCoord.xy);
  vec4 world=texelFetch(positionTex,pixel,0);
  vec3 normal=texelFetch(normalTex,pixel,0).xyz;
  vec2 mask=vec2(1);
  if(useShadow && world.w!=0.0) {
    for(int i=0;i<2;i++) {
      vec3 l=lightDir[i].xyz;
      if(lightDir[i].w<=0.0 || dot(normal,l)<=0.0) continue;
      float angle=randomValue(gl_FragCoord.xy,float(i)*2.0)*6.283185;
      float radius=sqrt(randomValue(gl_FragCoord.xy,float(i)*2.0+1.0))*tan(lightColor[i].w);
      vec3 ray=normalize(l+basis(l)*vec3(cos(angle)*radius,sin(angle)*radius,0));
      vec3 hit, hn;
      if(dot(normal,ray)>0.0) mask[i]=trace(world.xyz+normal*0.05,ray,1000.0,hit,hn)<0?1.0:0.0;
    }
  }
  color=vec4(mask,0,1);
}`;
const glLightingFragment = glLightingBindings + `
void main() {
  ivec2 pixel=ivec2(gl_FragCoord.xy);
  vec4 position=texelFetch(positionTex,pixel,0);
  if(position.w==0.0) {
    float t=size.y>1.0?(size.y-1.0-float(pixel.y))/(size.y-1.0):0.0;
    color=vec4(mix(skyTop,skyHorizon,t),1);return;
  }
  vec3 world=position.xyz;
  vec4 normalAO=texelFetch(normalTex,pixel,0);
  vec3 normal=normalAO.xyz;
  float ao=normalAO.w;
  int mat=int(position.w)-1;
  float vis=(useAO?ao:cachedAmbient(world,normal))*(sparseInfo.x>0.5&&useAO?cachedAmbient(world,normal):1.0);
  vec3 base=albedo[mat], params=materialParams[mat].xyz;
  if(cathedral) {
    vec3 cell=floor((world-normal*0.002)*24.0);
    float grain=fract(sin(dot(cell,vec3(12.9898,78.233,37.719)))*43758.5453);
    base*=0.86+0.22*grain;
    if(mat==3) base*=0.80+0.25*sin(world.x*93.0+sin(world.z*7.0));
  }
  vec3 result=base*(skyIncoming(normal)*vis+materialParams[mat].w);
  for(int i=0;i<6;i++) {
    vec3 delta=pointPosition[i].xyz-world;
    float d2=dot(delta,delta),r2=pointColor[i].w*pointColor[i].w;
    if(pointPosition[i].w<=0.0 || d2>=r2 || d2<1e-8) continue;
    float distance=sqrt(d2);vec3 direction=delta/distance;
    float cosine=max(dot(normal,direction),0.0);if(cosine<=0.0) continue;
    vec3 hit,hn;
    if(useShadow && trace(world+normal*0.04,direction,max(0.0,distance-0.08),hit,hn)>=0) continue;
    float fade=1.0-d2/r2;
    result+=base*pointColor[i].rgb*(cosine*pointPosition[i].w*fade*fade/(1.0+d2*1.8));
  }
  vec3 v=normalize(camera-world);
  for(int i=0;i<2;i++) {
    vec3 l=lightDir[i].xyz;
    float ndl=max(dot(normal,l),0.0), intensity=lightDir[i].w;
    if(ndl<=0.0||intensity<=0.0) continue;
    float visible=texelFetch(shadowTex,pixel,0)[i];
    vec3 h=normalize(l+v);
    float fresnel=params.y+(1.0-params.y)*pow(1.0-max(dot(v,h),0.0),5.0);
    float spec=(params.z+8.0)/(8.0*3.14159265)*pow(max(dot(normal,h),0.0),params.z)*fresnel*ndl;
    result+=(base*params.x*(1.0-fresnel)*ndl*intensity+vec3(clamp(spec*intensity,0.0,1.0)))*lightColor[i].rgb*visible;
  }
  if(useGI) {
    vec3 p=world, norm=normal, throughput=base, indirect=vec3(0);
    for(int bounce=0;bounce<2;bounce++) {
      float u=randomValue(gl_FragCoord.xy,10.0+float(bounce)*2.0);
      float angle=randomValue(gl_FragCoord.xy,11.0+float(bounce)*2.0)*6.283185;
      float cosine=sqrt(1.0-u);
      vec3 ray=basis(norm)*vec3(sqrt(u)*cos(angle),sqrt(u)*sin(angle),cosine);
      vec3 hit, hn;
      int material=trace(p+norm*0.04,ray,cathedral?1000.0:12.0,hit,hn);
      if(material<0) {
        if(cathedral) indirect+=throughput*skyIncoming(ray)*cosine;
        break;
      }
      vec3 incoming=skyIncoming(hn),hitOrigin=hit+hn*0.04;
      incoming*=cachedAmbient(hit,hn);
      incoming+=vec3(materialParams[material].w);
      for(int i=0;i<2;i++) {
        float nl=max(dot(hn,lightDir[i].xyz),0.0);vec3 hp,np;
        if(nl>0.0 && lightDir[i].w>0.0 && trace(hitOrigin,lightDir[i].xyz,1000.0,hp,np)<0)
          incoming+=lightColor[i].rgb*lightDir[i].w*nl;
      }
      throughput*=albedo[material]*cosine;
      indirect+=incoming*throughput;
      p=hit; norm=hn;
    }
    result+=clamp(indirect,0.0,4.0)*min(1.0,vis+0.15);
  }
  if(cathedral && (mat==4 || mat==11) && normal.y>0.9) {
    float a=randomValue(gl_FragCoord.xy,51.0)*6.283185;
    float r=sqrt(randomValue(gl_FragCoord.xy,52.0))*0.07;
    vec3 reflection=reflect(-v,normal);
    vec3 ray=normalize(reflection+basis(reflection)*vec3(cos(a)*r,sin(a)*r,0));
    ray=normalize(ray+normal*max(0.0,0.002-dot(ray,normal)));
    vec3 hit,hn;int m=trace(world+normal*0.04,ray,1000.0,hit,hn);
    vec3 reflected=skyHorizon;
    if(m>=0) {
      reflected=albedo[m]*(skyIncoming(hn)*cachedAmbient(hit,hn)+materialParams[m].w);
      for(int i=0;i<2;i++) {
        float ndl=max(dot(hn,lightDir[i].xyz),0.0);
        if(ndl>0.0 && lightDir[i].w>0.0) {
          vec3 hp,np;
          if(trace(hit+hn*0.05,lightDir[i].xyz,1000.0,hp,np)<0)
            reflected+=albedo[m]*lightColor[i].xyz*lightDir[i].w*ndl;
        }
      }
    }
    float fresnel=0.10+0.45*pow(1.0-max(dot(normal,v),0.0),5.0);
    result=mix(result,reflected,fresnel);
  }
  color=vec4(result,1);
}`;

export class WebGLRenderer {
  constructor(canvas, scene) {
    this.canvas = canvas;
    const gl = this.gl = canvas.getContext('webgl2', {antialias: false, alpha: false});
    if (!gl) throw new Error('浏览器未提供 WebGL2');
    if (!gl.getExtension('EXT_color_buffer_float')) throw new Error('完整 HDR TAA 需要 EXT_color_buffer_float');
    this.lost = false;
    canvas.addEventListener('webglcontextlost', event => { event.preventDefault(); this.lost = true; });
    this.passes = {
      geometry: this.program(scene.info[176]?compactMeshVertex:meshVertex, glGeometryFragment),
      shadow: this.program(fullVertex, glRawShadow),
      repair: this.program(fullVertex, glRepairFragment),
      filter: this.program(fullVertex, glShadowFilter),
      lighting: this.program(fullVertex, glLightingFragment),
      taa: this.program(fullVertex, glTaaResolve),
      present: this.program(fullVertex, glPostPresent),
      bloomExtract: this.program(fullVertex, glBloomExtract),
      bloomX: this.program(fullVertex, glBloomBlur(true)),
      bloomY: this.program(fullVertex, glBloomBlur(false)),
    };
    this.programs = Object.values(this.passes); this.locations = new Map();
    this.frame = 0; this.valid = false; this.queries = []; this.gpuMs = null;
    this.timer = gl.getExtension('EXT_disjoint_timer_query_webgl2');
    const {info, mesh, voxels} = scene;
    this.info = info;
    this.vao = gl.createVertexArray(); gl.bindVertexArray(this.vao);
    this.meshBuffer = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, this.meshBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, mesh, gl.STATIC_DRAW);
    for (const [index, count, offset] of (info[176]?[[0,3,0],[1,3,3],[2,3,6],[3,2,9],[4,4,11]]:[[0,3,0], [1,3,3], [2,1,6], [3,1,7]])) {
      gl.enableVertexAttribArray(index); gl.vertexAttribPointer(index, count, gl.FLOAT, false, info[176]?60:32, offset * 4);
      if(info[176]) gl.vertexAttribDivisor(index,1);
    }
    this.vertices = mesh.length / (info[176]?15:8); gl.bindVertexArray(null);
    this.voxels = gl.createTexture(); gl.bindTexture(gl.TEXTURE_3D, this.voxels);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
    gl.texImage3D(gl.TEXTURE_3D, 0, gl.R8UI, info[176]?1:info[2], info[176]?1:info[3], info[176]?1:info[2], 0, gl.RED_INTEGER, gl.UNSIGNED_BYTE, info[176]?new Uint8Array(1):voxels);
    for (const p of [gl.TEXTURE_MIN_FILTER, gl.TEXTURE_MAG_FILTER]) gl.texParameteri(gl.TEXTURE_3D, p, gl.NEAREST);
    for (const p of [gl.TEXTURE_WRAP_S, gl.TEXTURE_WRAP_T, gl.TEXTURE_WRAP_R]) gl.texParameteri(gl.TEXTURE_3D, p, gl.CLAMP_TO_EDGE);
    this.sparseVoxels=gl.createTexture();gl.bindTexture(gl.TEXTURE_2D,this.sparseVoxels);
    this.sparseTextureWidth=info[176]?Math.min(4096,gl.getParameter(gl.MAX_TEXTURE_SIZE)):1;
    const rows=info[176]?Math.ceil(voxels.byteLength/4/this.sparseTextureWidth):1;
    const packed=new Uint32Array(this.sparseTextureWidth*rows);
    if(info[176]) new Uint8Array(packed.buffer).set(voxels);
    gl.texImage2D(gl.TEXTURE_2D,0,gl.R32UI,this.sparseTextureWidth,rows,0,gl.RED_INTEGER,gl.UNSIGNED_INT,packed);
    gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MIN_FILTER,gl.NEAREST);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MAG_FILTER,gl.NEAREST);
    this.targets = [];
  }
  program(vertex, fragment) {
    const gl = this.gl, program = gl.createProgram();
    for (const [type, source] of [[gl.VERTEX_SHADER, vertex], [gl.FRAGMENT_SHADER, fragment]]) {
      const shader = gl.createShader(type); gl.shaderSource(shader, source); gl.compileShader(shader);
      if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(shader));
      gl.attachShader(program, shader); gl.deleteShader(shader);
    }
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(program));
    return program;
  }
  uniform(program, name, method, ...values) {
    const key = this.programs.indexOf(program) + ':' + name;
    if (!this.locations.has(key)) this.locations.set(key, this.gl.getUniformLocation(program, name));
    const location = this.locations.get(key);
    if (location !== null) this.gl[method](location, ...values);
  }
  reset() {
    this.valid = false; this.gpuMs = null;
    for (const query of this.queries) this.gl.deleteQuery(query);
    this.queries = [];
  }
  dispose() {
    this.reset();
    const gl = this.gl;
    for (const target of this.targets) { gl.deleteTexture(target.texture); gl.deleteFramebuffer(target.fbo); }
    for (const program of this.programs) gl.deleteProgram(program);
    gl.deleteTexture(this.voxels); gl.deleteTexture(this.sparseVoxels);
    gl.deleteBuffer(this.meshBuffer); gl.deleteVertexArray(this.vao);
    if (this.depth) gl.deleteRenderbuffer(this.depth);
    if (this.geometryFbo) gl.deleteFramebuffer(this.geometryFbo);
    gl.getExtension('WEBGL_lose_context')?.loseContext();
  }
  resize(width, height) {
    const gl = this.gl;
    if (this.targets.length && this.canvas.width === width && this.canvas.height === height) return;
    for (const target of this.targets) { gl.deleteTexture(target.texture); gl.deleteFramebuffer(target.fbo); }
    if (this.depth) gl.deleteRenderbuffer(this.depth);
    if (this.geometryFbo) gl.deleteFramebuffer(this.geometryFbo);
    this.canvas.width = width; this.canvas.height = height; this.targets = []; this.reset();
    const target = (format = gl.RGBA32F, w=width, h=height) => {
      const texture = gl.createTexture(); gl.bindTexture(gl.TEXTURE_2D, texture);
      gl.texImage2D(gl.TEXTURE_2D, 0, format, w, h, 0, gl.RGBA, format === gl.RGBA8 ? gl.UNSIGNED_BYTE : gl.FLOAT, null);
      for (const p of [gl.TEXTURE_MIN_FILTER, gl.TEXTURE_MAG_FILTER]) gl.texParameteri(gl.TEXTURE_2D, p, gl.NEAREST);
      for (const p of [gl.TEXTURE_WRAP_S, gl.TEXTURE_WRAP_T]) gl.texParameteri(gl.TEXTURE_2D, p, gl.CLAMP_TO_EDGE);
      const fbo = gl.createFramebuffer(); gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0);
      if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE) throw new Error('HDR framebuffer incomplete');
      const value = {texture, fbo, width:w, height:h}; this.targets.push(value); return value;
    };
    this.position = target(); this.normal = target(gl.RGBA16F);
    this.rasterCoverage=target(gl.RGBA32F,this.info[62]?width:1,this.info[62]?height:1);
    this.previousPosition=target();this.previousNormal=target(gl.RGBA16F);
    this.rawShadow = target(); this.filteredShadow = target(); this.linear = target();
    this.history = [target(), target()]; this.output = target(gl.RGBA8);
    this.bloom=[target(gl.RGBA32F,Math.ceil(width/2),Math.ceil(height/2)),target(gl.RGBA32F,Math.ceil(width/2),Math.ceil(height/2))];
    this.geometryFbo = gl.createFramebuffer(); gl.bindFramebuffer(gl.FRAMEBUFFER, this.geometryFbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this.position.texture, 0);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT1, gl.TEXTURE_2D, this.normal.texture, 0);
    gl.drawBuffers([gl.COLOR_ATTACHMENT0, gl.COLOR_ATTACHMENT1]);
    this.depth = gl.createRenderbuffer(); gl.bindRenderbuffer(gl.RENDERBUFFER, this.depth);
    gl.renderbufferStorage(gl.RENDERBUFFER, gl.DEPTH_COMPONENT24, width, height);
    gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.RENDERBUFFER, this.depth);
    if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE) throw new Error('G-buffer incomplete');
  }
  use(program, data, settings) {
    const gl = this.gl, s = data.state;
    gl.useProgram(program);
    const set = (name, method, ...v) => this.uniform(program, name, method, ...v);
    set('view', 'uniformMatrix4fv', false, s.subarray(0,16));
    set('previousVP', 'uniformMatrix4fv', false, s.subarray(49,65));
    set('size', 'uniform2f', data.width, data.height);
    set('jitter', 'uniform2fv', s.subarray(65,67));
    set('previousJitter','uniform2fv',data.previousJitter || this.lastState?.subarray(65,67) || [0,0]);
    set('cameraTranslated','uniform1i',data.cameraTranslated ?? (this.lastState?Math.hypot(s[16]-this.lastState[16],s[17]-this.lastState[17],s[18]-this.lastState[18])>1e-5:false));
    set('taaParams', 'uniform4f', s[68], s[74], this.valid ? 1 : 0, s[67]);
    set('filterParams', 'uniform4fv', s.subarray(69,73));
    set('taaEnabled', 'uniform1i', s[73]); set('filterEnabled', 'uniform1i', s[75]);
    set('camera', 'uniform3fv', s.subarray(16,19));
    set('verticalShift','uniform1f',s[77] || 0);
    set('skyTop', 'uniform3fv', s.subarray(19,22)); set('skyHorizon', 'uniform3fv', s.subarray(22,25));
    set('ambientTop', 'uniform3fv', s.subarray(25,28)); set('ground', 'uniform3fv', s.subarray(28,31));
    set('skyScale', 'uniform1f', s[31]); set('exposure', 'uniform1f', s[32]); set('frame', 'uniform1f', s[76]);
    set('origin', 'uniform3fv', this.info.subarray(4,7)); set('blockSize', 'uniform1f', this.info[7]);
    set('gridSize', 'uniform3i', this.info[2], this.info[3], this.info[2]);
    for (let i = 0; i < 2; i++) {
      set(`lightDir[${i}]`, 'uniform4fv', s.subarray(33+i*8,37+i*8));
      set(`lightColor[${i}]`, 'uniform4fv', s.subarray(37+i*8,41+i*8));
    }
    for (let i = 0; i < 16; i++) {
      const o=i<8?8+i*6:this.info.length>=127?63+(i-8)*6:8;
      set(`albedo[${i}]`, 'uniform3fv', this.info.subarray(o,o+3));
      set(`materialParams[${i}]`, 'uniform4f', this.info[o+3],this.info[o+4],this.info[o+5],this.info[111+i] || 0);
    }
    set('cathedral','uniform1i',this.info[62] || 0);
    set('sparseInfo','uniform3f',this.info[176]||0,this.info[177]||0,this.info[178]||0);
    set('sparseTextureWidth','uniform1i',this.sparseTextureWidth);
    gl.activeTexture(gl.TEXTURE7);gl.bindTexture(gl.TEXTURE_2D,this.sparseVoxels);set('sparseVoxels','uniform1i',7);
    for(let i=0;i<6;i++) {
      const o=128+i*8;
      if(i<(this.info[127]||0)) {
        set(`pointPosition[${i}]`,'uniform4fv',this.info.subarray(o,o+4));
        set(`pointColor[${i}]`,'uniform4fv',this.info.subarray(o+4,o+8));
      } else {
        set(`pointPosition[${i}]`,'uniform4f',0,0,0,0);set(`pointColor[${i}]`,'uniform4f',0,0,0,0);
      }
    }
    set('useAO', 'uniform1i', settings.ao); set('useShadow', 'uniform1i', settings.shadows); set('useGI', 'uniform1i', settings.gi);
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_3D, this.voxels); set('voxels', 'uniform1i', 0);
  }
  texture(program, name, unit, target) {
    const gl = this.gl;
    gl.activeTexture(gl.TEXTURE0 + unit); gl.bindTexture(gl.TEXTURE_2D, target.texture);
    this.uniform(program, name, 'uniform1i', unit);
  }
  fullscreen(program, target, data, settings, source, history) {
    const gl = this.gl;
    gl.bindFramebuffer(gl.FRAMEBUFFER, target.fbo);gl.viewport(0,0,target.width??data.width,target.height??data.height); this.use(program, data, settings);
    this.texture(program, 'positionTex', 1, this.position);
    this.texture(program, 'normalTex', 2, this.normal);
    this.texture(program,'previousPositionTex',8,this.previousPosition);
    this.texture(program,'previousNormalTex',9,this.previousNormal);
    if (source) this.texture(program, 'sourceTex', 3, source);
    if (history) this.texture(program, 'historyTex', 4, history);
    this.texture(program, 'shadowTex', 5, this.filteredShadow);
    this.texture(program, 'bloomTex', 6, this.bloom[0]);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
  }
  pollTimers() {
    if (!this.timer) return;
    const gl = this.gl;
    if (gl.getParameter(this.timer.GPU_DISJOINT_EXT)) {
      for (const q of this.queries) gl.deleteQuery(q);
      this.queries = []; this.gpuMs = null; return;
    }
    while (this.queries.length && gl.getQueryParameter(this.queries[0], gl.QUERY_RESULT_AVAILABLE)) {
      const q = this.queries.shift(); this.gpuMs = gl.getQueryParameter(q, gl.QUERY_RESULT) / 1e6; gl.deleteQuery(q);
      this.gpuSampleVersion = (this.gpuSampleVersion ?? 0) + 1;
    }
  }
  draw(data, settings) {
    if (this.lost) throw new Error('WebGL 上下文已丢失');
    const gl = this.gl, start = performance.now();
    this.resize(data.width, data.height); this.pollTimers();
    let query = null;
    if (this.timer && this.queries.length < 4) { query = gl.createQuery(); gl.beginQuery(this.timer.TIME_ELAPSED_EXT, query); }
    gl.viewport(0,0,data.width,data.height); gl.bindFramebuffer(gl.FRAMEBUFFER,this.geometryFbo);
    gl.clearColor(0,0,0,0); gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT); gl.enable(gl.DEPTH_TEST);
    gl.bindVertexArray(this.vao); this.use(this.passes.geometry,data,settings);
    if(this.info[176]) gl.drawArraysInstanced(gl.TRIANGLES,0,6,this.vertices);else gl.drawArrays(gl.TRIANGLES,0,this.vertices); gl.bindVertexArray(null); gl.disable(gl.DEPTH_TEST);
    if(this.info[62]) {
      gl.bindFramebuffer(gl.READ_FRAMEBUFFER,this.position.fbo);gl.bindFramebuffer(gl.DRAW_FRAMEBUFFER,this.rasterCoverage.fbo);
      gl.blitFramebuffer(0,0,data.width,data.height,0,0,data.width,data.height,gl.COLOR_BUFFER_BIT,gl.NEAREST);
      gl.bindFramebuffer(gl.FRAMEBUFFER,this.geometryFbo);this.use(this.passes.repair,data,settings);
      this.texture(this.passes.repair,'coverageTex',1,this.rasterCoverage);gl.drawArrays(gl.TRIANGLES,0,3);
    }
    this.fullscreen(this.passes.shadow,this.rawShadow,data,settings);
    this.fullscreen(this.passes.filter,this.filteredShadow,data,settings,this.rawShadow);
    this.fullscreen(this.passes.lighting,this.linear,data,settings);
    const index = this.frame % 2;
    this.fullscreen(this.passes.taa,this.history[index],data,settings,this.linear,this.history[1-index]);
    if(this.info[62]) {
      this.fullscreen(this.passes.bloomExtract,this.bloom[0],data,settings,this.history[index]);
      this.fullscreen(this.passes.bloomX,this.bloom[1],data,settings,this.bloom[0]);
      this.fullscreen(this.passes.bloomY,this.bloom[0],data,settings,this.bloom[1]);
    }
    this.fullscreen(this.passes.present,this.output,data,settings,this.history[index]);
    gl.bindFramebuffer(gl.READ_FRAMEBUFFER,this.output.fbo); gl.bindFramebuffer(gl.DRAW_FRAMEBUFFER,null);
    gl.blitFramebuffer(0,0,data.width,data.height,0,0,data.width,data.height,gl.COLOR_BUFFER_BIT,gl.NEAREST);
    for(const [from,to] of [[this.position,this.previousPosition],[this.normal,this.previousNormal]]) {
      gl.bindFramebuffer(gl.READ_FRAMEBUFFER,from.fbo);gl.bindFramebuffer(gl.DRAW_FRAMEBUFFER,to.fbo);
      gl.blitFramebuffer(0,0,data.width,data.height,0,0,data.width,data.height,gl.COLOR_BUFFER_BIT,gl.NEAREST);
    }
    if (query) { gl.endQuery(this.timer.TIME_ELAPSED_EXT); this.queries.push(query); }
    this.lastOutput = this.output; this.lastHistory = this.history[index]; this.lastState = data.state.slice();
    this.valid = Boolean(data.state[73]); this.frame++;
    return {submitMs:performance.now()-start,gpuMs:this.gpuMs};
  }
}
