// Scene adapted from franky-adl/3d-wave-grid. See LICENSE.txt.
import * as THREE from 'three';
import { gsap } from 'gsap';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { ShaderPass } from 'three/addons/postprocessing/ShaderPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';
import { VignetteRGBShiftShader } from './VignetteRGBShiftShader.js';
import Stage from './Stage.js';

export function createWaveGrid(host) {
    const canvas = host.querySelector('canvas');
    let renderer;
    try {
        renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
    } catch {
        host.dataset.fallback = 'true';
        return () => {};
    }
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.95;
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFShadowMap;

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(40, 1, 0.1, 200);
    camera.up.set(0, 0, -1);
    const stage = new Stage(scene, camera, host);
    const composer = new EffectComposer(renderer);
    const renderPass = new RenderPass(scene, camera);
    const colorPass = new ShaderPass(VignetteRGBShiftShader);
    const outputPass = new OutputPass();
    composer.addPass(renderPass);
    composer.addPass(colorPass);
    composer.addPass(outputPass);
    const pointer = new THREE.Vector2();
    const smoothed = new THREE.Vector2();
    const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)');
    const panel = host.closest('.transition-slide');
    let intersecting = false;
    let running = false;
    let lost = false;

    function updateCamera() {
        const alpha = smoothed.y * Math.PI * 0.03;
        const beta = smoothed.x * Math.PI * 0.05;
        camera.position.set(-12 * Math.cos(alpha) * Math.sin(beta), 12 * Math.cos(alpha) * Math.cos(beta), 12 * Math.sin(alpha));
        camera.lookAt(0, 0, 0);
        camera.updateMatrixWorld();
    }
    function render() {
        if (lost) return;
        composer.render();
        host.dataset.ready = 'true';
    }
    function tick(_time, deltaMs) {
        const delta = Math.min(deltaMs / 1000, 0.05);
        smoothed.lerp(pointer, 1 - Math.exp(-2.45 * delta));
        updateCamera();
        stage.update(delta);
        render();
    }
    function sync() {
        const visible = intersecting && panel?.dataset.visible !== 'false' && !document.hidden && !lost;
        const shouldRun = visible && !reducedMotion.matches;
        if (shouldRun !== running) {
            running = shouldRun;
            if (running) gsap.ticker.add(tick);
            else gsap.ticker.remove(tick);
        }
        // A still rendering preserves the background when motion is reduced.
        if (visible && reducedMotion.matches) {
            smoothed.set(0, 0);
            updateCamera();
            render();
        }
    }
    function resize() {
        const { width, height } = host.getBoundingClientRect();
        if (!width || !height || lost) return;
        const pixelRatio = Math.min(devicePixelRatio, 1.5);
        renderer.setPixelRatio(pixelRatio);
        renderer.setSize(width, height, false);
        composer.setPixelRatio(pixelRatio);
        composer.setSize(width, height);
        camera.aspect = width / height;
        camera.updateProjectionMatrix();
        updateCamera();
        if (intersecting) render();
    }
    function onPointer(event) {
        if (reducedMotion.matches || !running) return;
        const rect = host.getBoundingClientRect();
        pointer.set((event.clientX - rect.left) / rect.width * 2 - 1, 1 - (event.clientY - rect.top) / rect.height * 2);
    }
    function onLeave() {
        pointer.set(0, 0);
        stage.mouseTrail.lastPoint = null;
    }
    function onContextLost(event) {
        event.preventDefault();
        lost = true;
        host.dataset.fallback = 'true';
        sync();
    }
    function onContextRestored() {
        lost = false;
        delete host.dataset.fallback;
        resize();
        sync();
    }
    const intersection = new IntersectionObserver(([entry]) => {
        intersecting = entry.isIntersecting;
        sync();
    });
    intersection.observe(host);
    const visibility = new MutationObserver(sync);
    if (panel) visibility.observe(panel, { attributes: true, attributeFilter: ['data-visible'] });
    const size = new ResizeObserver(resize);
    size.observe(host);
    host.addEventListener('pointermove', onPointer);
    host.addEventListener('pointerleave', onLeave);
    canvas.addEventListener('webglcontextlost', onContextLost);
    canvas.addEventListener('webglcontextrestored', onContextRestored);
    document.addEventListener('visibilitychange', sync);
    reducedMotion.addEventListener('change', sync);
    updateCamera();
    resize();

    return () => {
        gsap.ticker.remove(tick);
        intersection.disconnect();
        visibility.disconnect();
        size.disconnect();
        host.removeEventListener('pointermove', onPointer);
        host.removeEventListener('pointerleave', onLeave);
        canvas.removeEventListener('webglcontextlost', onContextLost);
        canvas.removeEventListener('webglcontextrestored', onContextRestored);
        document.removeEventListener('visibilitychange', sync);
        reducedMotion.removeEventListener('change', sync);
        stage.mouseTrail.dispose();
        stage.instancedMesh.geometry.dispose();
        stage.instancedMesh.material.dispose();
        stage.instancedMesh.customDepthMaterial.dispose();
        stage.instancedMesh.dispose();
        stage.directionalLight.shadow.dispose();
        colorPass.dispose();
        outputPass.dispose();
        composer.dispose();
        renderer.dispose();
    };
}
