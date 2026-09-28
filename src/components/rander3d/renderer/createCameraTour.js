import { gsap } from 'gsap';
import { ScrollTrigger } from 'gsap/ScrollTrigger';
import { Observer } from 'gsap/Observer';
import { ScrollToPlugin } from 'gsap/ScrollToPlugin';
import { cameraAnchors, cameraAtProgress } from './cameraAnchors.js';

gsap.registerPlugin(ScrollTrigger, Observer, ScrollToPlugin);

export function createCameraTour(root) {
    const caption = root.querySelector('[data-tour-caption]');
    const title = root.querySelector('[data-tour-title]');
    const body = root.querySelector('[data-tour-body]');
    const controls = root.querySelector('[data-controls]');
    const lastIndex = cameraAnchors.length - 1;
    const media = gsap.matchMedia();
    let camera = cameraAtProgress(0), renderedCamera;
    let ready = false, stopped = false, disposed = false;
    let sync = () => {}, reveal = () => {}, release = () => {};

    // Run after the surrounding section's GSAP handoff has installed its sticky layout.
    const setupFrame = requestAnimationFrame(() => {
        media.add({ reduced: '(prefers-reduced-motion: reduce)', normal: '(prefers-reduced-motion: no-preference)' }, (context) => {
            const reduced = context.conditions.reduced;
            const handoff = root.closest('[data-handoff-next]');
            const sharedPin = handoff?.closest('[data-handoff-animated]');
            const runway = handoff?.querySelector('[data-handoff-runway]');
            if (sharedPin) gsap.set(handoff, { '--handoff-tail-height': `${lastIndex * 100}svh` });

            let trigger, observer, scrollTween, resizeCall;
            let moving = false, gestureLocked = false, targetIndex = 0, shownIndex = -1;
            let exiting = false, refreshIndex = null;
            const events = new AbortController();
            gsap.set(caption, { autoAlpha: 0, y: 12 });
            root.dataset.tourMoving = 'false';
            root.dataset.tourIndex = '0';

            const anchorScroll = (index) => trigger.start + 1 + (trigger.end - trigger.start - 2) * index / lastIndex;
            const progress = () => gsap.utils.clamp(0, lastIndex,
                (trigger.scroll() - trigger.start - 1) / (trigger.end - trigger.start - 2) * lastIndex);
            const canCapture = () => ready && !stopped && !disposed && trigger?.isActive && !exiting;

            function hideCaption() {
                if (shownIndex === -1) return;
                shownIndex = -1;
                caption.setAttribute('aria-hidden', 'true');
                gsap.to(caption, { autoAlpha: 0, y: 12, duration: reduced ? 0 : 0.18, overwrite: true });
            }

            reveal = () => {
                if (!canCapture() || moving || !renderedCamera || shownIndex === targetIndex) return;
                if (Math.abs(progress() - targetIndex) > 0.002) return;
                const anchor = cameraAnchors[targetIndex];
                if (!Object.keys(anchor.camera).every((key) => Math.abs(anchor.camera[key] - renderedCamera[key]) < 0.003)) return;
                title.textContent = anchor.title;
                body.textContent = anchor.text;
                caption.setAttribute('aria-hidden', 'false');
                root.dataset.tourIndex = String(targetIndex);
                shownIndex = targetIndex;
                gsap.fromTo(caption, { autoAlpha: 0, y: 12 }, { autoAlpha: 1, y: 0, duration: reduced ? 0 : 0.4, overwrite: true });
            };

            function travel(index, entering = false) {
                if (!trigger || disposed || stopped) return;
                scrollTween?.kill();
                hideCaption();
                targetIndex = index;
                moving = true;
                root.dataset.tourMoving = 'true';
                const destination = anchorScroll(index);
                scrollTween = gsap.to(window, {
                    paused: true,
                    scrollTo: { y: destination, autoKill: false },
                    duration: reduced ? 0 : entering ? 0.55 : 1.8,
                    ease: 'power2.inOut',
                    onUpdate: () => ScrollTrigger.update(),
                    onComplete: () => {
                        scrollTween = null;
                        moving = false;
                        root.dataset.tourMoving = 'false';
                        camera = cameraAtProgress(index);
                        reveal();
                    },
                });
                scrollTween.play();
            }

            function leave(direction) {
                exiting = true;
                observer.disable();
                hideCaption();
                const destination = direction > 0
                    ? trigger.end + root.offsetHeight * 0.8
                    : trigger.start - root.offsetHeight * 0.6;
                scrollTween = gsap.to(window, {
                    paused: true,
                    scrollTo: { y: Math.max(0, destination), autoKill: true },
                    duration: reduced ? 0 : 0.75,
                    ease: 'power2.inOut',
                    onUpdate: () => ScrollTrigger.update(),
                    onComplete: () => { scrollTween = null; exiting = false; sync(); },
                    onInterrupt: () => { scrollTween = null; exiting = false; sync(); },
                });
                scrollTween.play();
            }

            function step(direction) {
                if (!canCapture() || !direction) return;
                if (moving || gestureLocked) { gestureLocked = true; return; }
                gestureLocked = true;
                const next = targetIndex + direction;
                if (next < 0 || next > lastIndex) leave(direction);
                else travel(next);
            }

            observer = Observer.create({
                target: window,
                type: 'wheel,touch',
                preventDefault: true,
                allowClicks: true,
                lockAxis: true,
                tolerance: 12,
                onStopDelay: 0.25,
                ignoreCheck: (event) => event.ctrlKey || event.metaKey || controls.contains(event.target),
                onChangeY: (self) => step((self.event.type === 'wheel' ? 1 : -1) * Math.sign(self.deltaY)),
                onStop: () => { gestureLocked = false; },
            });
            observer.disable();

            sync = (entryIndex) => {
                if (!canCapture()) { observer.disable(); hideCaption(); return; }
                if (!observer.isEnabled) {
                    observer.enable();
                    gestureLocked = false;
                    if (!moving) travel(entryIndex ?? Math.round(progress()), true);
                }
            };

            trigger = ScrollTrigger.create({
                trigger: sharedPin ? handoff : root,
                start: () => sharedPin ? `top+=${runway.offsetHeight} top` : 'top top',
                end: () => `+=${root.offsetHeight * lastIndex}`,
                pin: sharedPin ? false : root,
                pinSpacing: !sharedPin,
                invalidateOnRefresh: true,
                onUpdate: (self) => {
                    trigger = self;
                    const value = progress();
                    camera = cameraAtProgress(value);
                    if (Math.abs(value - targetIndex) > 0.002) hideCaption();
                },
                onToggle: (self) => { if (!self.isActive) sync(); },
                onEnter: () => sync(0),
                onEnterBack: () => sync(lastIndex),
                onRefreshInit: (self) => {
                    refreshIndex = ready && (self.isActive || moving) && !exiting ? targetIndex : null;
                },
                onRefresh: (self) => {
                    trigger = self;
                    camera = cameraAtProgress(progress());
                    if (refreshIndex !== null && !exiting) {
                        const index = refreshIndex;
                        resizeCall?.kill();
                        resizeCall = gsap.delayedCall(0.05, () => travel(index, true));
                    }
                    sync();
                },
            });

            const settleScroll = () => {
                if (!canCapture() || moving || scrollTween) return;
                const nearest = Math.round(progress());
                if (Math.abs(progress() - nearest) > 0.002) travel(nearest, true);
                else { targetIndex = nearest; reveal(); }
            };
            ScrollTrigger.addEventListener('scrollEnd', settleScroll);

            window.addEventListener('keydown', (event) => {
                if (!canCapture() || event.ctrlKey || event.metaKey || event.altKey || event.target.closest?.('input, select, button, textarea, [contenteditable]')) return;
                const direction = ['ArrowDown', 'PageDown', ' '].includes(event.key) ? 1 : ['ArrowUp', 'PageUp'].includes(event.key) ? -1 : 0;
                if (!direction) return;
                event.preventDefault();
                if (!event.repeat) { gestureLocked = false; step(event.shiftKey && event.key === ' ' ? -1 : direction); }
            }, { signal: events.signal });

            release = () => {
                observer.disable();
                resizeCall?.kill();
                scrollTween?.kill();
                scrollTween = null;
                moving = false;
                root.dataset.tourMoving = 'false';
                hideCaption();
            };
            ScrollTrigger.refresh();

            return () => {
                release();
                observer.kill();
                trigger.kill();
                events.abort();
                ScrollTrigger.removeEventListener('scrollEnd', settleScroll);
                gsap.killTweensOf(caption);
                delete root.dataset.tourIndex;
                delete root.dataset.tourMoving;
                sync = reveal = release = () => {};
            };
        });
    });

    return {
        getCamera: () => ({ ...camera }),
        onFrame(pose) {
            renderedCamera = pose;
            if (!ready) { ready = true; sync(); }
            reveal();
        },
        stop() { stopped = true; ready = false; release(); },
        dispose() {
            disposed = true;
            cancelAnimationFrame(setupFrame);
            media.revert();
        },
    };
}
