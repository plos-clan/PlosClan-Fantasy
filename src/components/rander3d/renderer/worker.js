import createRenderer from './renderer.js';
import rendererWasmUrl from '../../../assets/rendertm/renderer.wasm?url';

let renderer;
let currentCamera;

function rememberCamera(state) {
  currentCamera = {
    x: state[16], y: state[17], z: state[18],
    yaw: Math.atan2(-state[8], state[0]),
    pitch: Math.atan2(-state[6], state[5]),
  };
}

function cameraInput(camera) {
  const values = ['x', 'y', 'z', 'yaw', 'pitch'].map(key => camera?.[key]);
  if (!values.every(Number.isFinite) || Math.abs(camera.pitch) > 1.4) throw new Error('Invalid camera pose');
  const dx = camera.x - currentCamera.x, dy = camera.y - currentCamera.y, dz = camera.z - currentCamera.z;
  const cy = Math.cos(camera.yaw), sy = Math.sin(camera.yaw);
  const forward = (dx * sy + dz * cy) / Math.cos(camera.pitch);
  // Invert Camera::move: yaw-only strafe, pitched forward, world-space vertical.
  // The existing bridge advances by dt * 8, so dt=0.1 gives a scale of 0.8.
  return [(dx * cy - dz * sy) / 0.8, (dy + forward * Math.sin(camera.pitch)) / 0.8, forward / 0.8,
    Math.atan2(Math.sin(camera.yaw - currentCamera.yaw), Math.cos(camera.yaw - currentCamera.yaw)),
    camera.pitch - currentCamera.pitch];
}
// Vite owns the asset URL; it need not share a filename or directory with the worker.
createRenderer({ locateFile: () => new URL(rendererWasmUrl, self.location.href).href }).then(module => {
  renderer = module;
  postMessage({type: 'ready'});
}).catch(error => postMessage({type: 'error', message: String(error)}));

self.onmessage = ({data}) => {
  try {
    if (!renderer) throw new Error('Renderer is not ready');
    if (data.type === 'scene') {
      const start = performance.now();
      const pointer = renderer._prepare_scene(data.preset === 'legacy' ? 0 : data.preset === 'minecraft' ? 1 : data.preset === 'cathedral' ? 2 : 3, data.seed ?? 1337);
      const info = new Float32Array(renderer.HEAPU8.buffer, pointer, renderer._scene_info_size()).slice();
      const mesh = new Float32Array(renderer.HEAPU8.buffer, renderer._scene_mesh(), info[0]).slice();
      const voxels = renderer.HEAPU8.slice(renderer._scene_voxels(), renderer._scene_voxels() + info[1]);
      const cameraPointer = renderer._gpu_frame(320, 180, 0, 0, 0, 0, 0, 0);
      rememberCamera(new Float32Array(renderer.HEAPU8.buffer, cameraPointer, 78));
      postMessage({type: 'scene', info, mesh, voxels, generationMs: performance.now()-start, seed: data.seed ?? 1337}, [info.buffer, mesh.buffer, voxels.buffer]);
    } else if (data.type === 'configure') {
      renderer._configure(...data.values);
    } else if (data.type === 'frame') {
      if (data.backend !== 'webgl' && data.backend !== 'webgpu') throw new Error(`Unsupported GPU backend: ${data.backend}`);
      const start = performance.now();
      const input = cameraInput(data.camera);
      const pointer = renderer._gpu_frame(data.width, data.height, 0.1, ...input);
      const state = new Float32Array(renderer.HEAPU8.buffer, pointer, 78).slice();
      rememberCamera(state);
      postMessage({type: 'frame', revision: data.revision, backend: data.backend, width: data.width, height: data.height,
        state, camera: currentCamera, moving: input.some(value => Math.abs(value) > 1e-5), renderMs: performance.now() - start,
        heapBytes: renderer.HEAPU8.byteLength}, [state.buffer]);
    }
  } catch (error) {
    postMessage({type: 'error', message: String(error)});
  }
};
