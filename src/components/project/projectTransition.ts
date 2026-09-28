import { gsap } from 'gsap';

type TransitionState = { phase: 'enter' | 'exit'; progress: number };
const states = new WeakMap<HTMLElement, TransitionState>();

export function notifyProjectTransition(panel: HTMLElement, state: TransitionState | null) {
    const previous = states.get(panel);
    if (state) {
        if (previous?.phase === state.phase && previous.progress === state.progress) return;
        states.set(panel, state);
    } else {
        states.delete(panel);
    }
    panel.dispatchEvent(new CustomEvent(`project:${state?.phase ?? 'reset'}`, { detail: state }));
}

export function registerProjectAnimation(
    selector: string,
    createTimeline: (root: HTMLElement) => gsap.core.Timeline,
) {
    let cleanup = () => {};
    function init() {
        cleanup();
        const disposers = Array.from(document.querySelectorAll<HTMLElement>(selector), (root) => {
            const panel = root.closest<HTMLElement>('.transition-slide');
            if (!panel) return () => {};
            let context: gsap.Context | undefined;
            let timeline: gsap.core.Timeline | undefined;
            const reset = () => {
                context?.revert();
                context = undefined;
                timeline = undefined;
            };
            const update = (state: TransitionState) => {
                if (!timeline) context = gsap.context(() => { timeline = createTimeline(root); }, root);
                timeline!.progress((state.phase === 'enter' ? state.progress : 1 + state.progress) / 2);
            };
            const onTransition = (event: Event) => update((event as CustomEvent<TransitionState>).detail);
            panel.addEventListener('project:enter', onTransition);
            panel.addEventListener('project:exit', onTransition);
            panel.addEventListener('project:reset', reset);
            const state = states.get(panel);
            if (state) update(state);
            return () => {
                panel.removeEventListener('project:enter', onTransition);
                panel.removeEventListener('project:exit', onTransition);
                panel.removeEventListener('project:reset', reset);
                reset();
            };
        });
        cleanup = () => disposers.forEach((dispose) => dispose());
    }
    init();
    document.addEventListener('astro:page-load', init);
    document.addEventListener('astro:before-swap', () => cleanup());
}
