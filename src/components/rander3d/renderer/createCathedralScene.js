import { createCameraTour } from './createCameraTour.js';

// Astro lifecycle adapter for RenderTM cathedral build 43940b72739d4e0c6da7.
const backendNames = { webgpu: 'WebGPU', webgl: 'WebGL2' };
const settingNames = ['ao', 'shadows', 'gi', 'taa', 'paused', 'shadowFilter', 'taaClamp'];

export function mountCathedralScene(root) {
    const find = (name) => root.querySelector(`[data-${name}]`);
    const stage = find('stage');
    const controls = find('controls');
    const backendLabel = find('backend-label');
    const resolution = find('resolution');
    const globalIllumination = root.querySelector('[data-setting="gi"]');
    const status = find('status');
    const placeholder = find('placeholder');
    const canvases = Object.fromEntries(Object.keys(backendNames).map((name) => [name, root.querySelector(`[data-canvas="${name}"]`)]));
    const renderers = {};
    const unavailableBackends = new Set();
    const tour = createCameraTour(root);
    const events = new AbortController();
    const on = (target, name, handler) => target.addEventListener(name, handler, { signal: events.signal });
    const settings = () => ({
        ao: true,
        shadows: true,
        gi: globalIllumination.checked,
        taa: true,
        paused: true,
        shadowFilter: true,
        taaClamp: true,
    });
    let worker, scene, canvas;
    let backend = null;
    let started = false, visible = false, ready = false, busy = false, disposed = false, failed = false;
    let animation = 0, revision = 0;
    let fallbackReason = '';

    function canRender() {
        return ready && visible && !document.hidden && !failed && !disposed;
    }

    function schedule() {
        if (!animation && canRender()) animation = requestAnimationFrame(tick);
    }

    function resetHistory() {
        revision++;
        for (const renderer of Object.values(renderers)) renderer.reset();
    }

    function configure() {
        if (!ready) return;
        resetHistory();
        const values = settings();
        worker.postMessage({ type: 'configure', values: settingNames.map((name) => Number(values[name])) });
        schedule();
    }

    function fail(error) {
        if (disposed || failed) return;
        failed = true;
        ready = busy = false;
        cancelAnimationFrame(animation);
        worker?.terminate();
        for (const name of Object.keys(renderers)) {
            renderers[name].dispose();
            delete renderers[name];
        }
        tour.stop();
        controls.disabled = true;
        backendLabel.textContent = '不可用';
        stage.setAttribute('aria-busy', 'false');
        placeholder.hidden = false;
        placeholder.textContent = '教堂场景加载失败，请刷新页面重试。';
        status.textContent = `加载或渲染失败：${error.message || error}`;
    }

    async function selectBackend(requested = 'webgpu', reason = '') {
        ready = false;
        controls.disabled = true;
        backendLabel.textContent = '初始化中';
        resetHistory();
        const candidates = requested === 'webgl' ? ['webgl', 'webgpu'] : ['webgpu', 'webgl'];
        for (const name of candidates) {
            if (disposed || failed) return;
            if (unavailableBackends.has(name)) continue;
            status.textContent = `正在初始化 ${backendNames[name]}…`;
            try {
                if (name === 'webgpu' && !renderers.webgpu) {
                    const { WebGPURenderer } = await import('./webgpu.js');
                    if (disposed || failed) return;
                    const renderer = await WebGPURenderer.create(canvases.webgpu, scene);
                    if (disposed || failed) { renderer.dispose(); return; }
                    renderers.webgpu = renderer;
                } else if (name === 'webgl' && !renderers.webgl) {
                    const { WebGLRenderer } = await import('./webgl.js');
                    if (disposed || failed) return;
                    renderers.webgl = new WebGLRenderer(canvases.webgl, scene);
                }
                canvas = canvases[name];
                for (const element of Object.values(canvases)) element.hidden = element !== canvas;
                backend = name;
                backendLabel.textContent = backendNames[name];
                fallbackReason = reason ? `${reason}，已回退 ${backendNames[name]}` : '';
                ready = true;
                controls.disabled = false;
                status.textContent = fallbackReason || `正在渲染 · ${backendNames[name]}`;
                configure();
                return;
            } catch {
                unavailableBackends.add(name);
                reason = `${backendNames[name]} 不可用`;
            }
        }
        fail('WebGPU 和 WebGL2 均不可用，请检查浏览器的硬件加速设置');
    }

    async function receive({ data }) {
        if (disposed || failed) return;
        if (data.type === 'error') return fail(data.message);
        if (data.type === 'ready') {
            status.textContent = '正在生成教堂与环境光遮蔽，首次加载可能需要几秒…';
            worker.postMessage({ type: 'scene', preset: 'cathedral', seed: 1337 });
        } else if (data.type === 'scene') {
            scene = data;
            await selectBackend();
        } else if (data.type === 'frame') {
            busy = false;
            // Settings and backend changes invalidate already queued frames.
            if (!canRender() || data.backend !== backend || data.revision !== revision) { schedule(); return; }
            try {
                renderers[data.backend].draw(data, settings());
            } catch {
                unavailableBackends.add(data.backend);
                renderers[data.backend].dispose();
                delete renderers[data.backend];
                await selectBackend(data.backend, `${backendNames[data.backend]} 已停止`);
                return;
            }
            placeholder.hidden = true;
            stage.setAttribute('aria-busy', 'false');
            status.textContent = fallbackReason || `正在渲染 · ${backendNames[data.backend]}`;
            tour.onFrame(data.camera);
            schedule();
        }
    }

    function tick() {
        animation = 0;
        if (!canRender()) return;
        if (busy || (backend === 'webgpu' && renderers.webgpu?.pendingFrames >= 2)) { schedule(); return; }
        const width = Number(resolution.value);
        busy = true;
        worker.postMessage({ type: 'frame', revision, backend, width, height: width * 9 / 16, camera: tour.getCamera() });
    }

    function start() {
        if (started || disposed) return;
        started = true;
        backendLabel.textContent = 'Loading';
        status.textContent = '正在加载教堂渲染器…';
        try {
            worker = new Worker(new URL('./worker.js', import.meta.url), { type: 'module' });
            worker.onerror = (event) => fail(event.message || '渲染线程加载失败');
            worker.onmessage = (event) => { receive(event).catch(fail); };
        } catch (error) { fail(error); }
    }

    on(resolution, 'change', configure);
    on(globalIllumination, 'change', configure);
    on(document, 'visibilitychange', schedule);

    const transitionPanel = root.closest('[data-handoff-content]');
    let inViewport = false;
    function syncVisibility() {
        const nextVisible = inViewport && !transitionPanel?.inert && transitionPanel?.dataset.visible !== 'false';
        if (visible === nextVisible) return;
        visible = nextVisible;
        if (visible) {
            start();
            schedule();
        } else {
            cancelAnimationFrame(animation);
            animation = 0;
        }
    }
    const observer = new IntersectionObserver(([entry]) => {
        inViewport = entry.isIntersecting;
        syncVisibility();
    });
    const panelObserver = new MutationObserver(syncVisibility);
    if (transitionPanel) panelObserver.observe(transitionPanel, { attributes: true, attributeFilter: ['inert', 'data-visible'] });
    observer.observe(stage);

    return () => {
        disposed = true;
        tour.dispose();
        observer.disconnect();
        panelObserver.disconnect();
        events.abort();
        cancelAnimationFrame(animation);
        worker?.terminate();
        for (const renderer of Object.values(renderers)) renderer.dispose();
        scene = null;
    };
}
