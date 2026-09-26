// @ts-check
import { defineConfig } from 'astro/config';

import tailwindcss from '@tailwindcss/vite';

// https://astro.build/config
export default defineConfig({
  vite: {
    plugins: [tailwindcss()],
    // Discover the lazy 3D scene's dependencies before serving the page.
    // Otherwise its first import can invalidate already-loaded modules in dev.
    optimizeDeps: {
      include: [
        'gsap',
        'gsap/ScrollTrigger',
        'three',
        'three/addons/postprocessing/EffectComposer.js',
        'three/addons/postprocessing/RenderPass.js',
        'three/addons/postprocessing/ShaderPass.js',
        'three/addons/postprocessing/OutputPass.js',
      ],
    },
  }
});
