/// <reference types="vitest/config" />
import { defineConfig, loadEnv } from 'vite';
import react from '@vitejs/plugin-react';

/**
 * Vite configuration.
 *
 * In development the app talks to a same-origin `/api` path, which is proxied to
 * the FastAPI server. That keeps CORS out of the picture locally and makes the
 * Server-Sent Events stream behave exactly as it does behind a production proxy.
 */
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '');
  const target = env.VITE_DEV_PROXY_TARGET ?? 'http://127.0.0.1:8000';

  return {
    plugins: [react()],
    server: {
      port: 5173,
      strictPort: true,
      proxy: {
        // The interactive API docs live on the backend too, so the footer link works in dev.
        '/docs': { target, changeOrigin: true },
        '/openapi.json': { target, changeOrigin: true },
        '/api': {
          target,
          changeOrigin: true,
          // SSE must not be buffered or timed out by the dev proxy.
          ws: false,
          configure: (proxy) => {
            proxy.on('proxyRes', (proxyRes) => {
              if (proxyRes.headers['content-type']?.includes('text/event-stream')) {
                proxyRes.headers['cache-control'] = 'no-cache, no-transform';
              }
            });
          },
        },
      },
    },
    preview: {
      port: 4173,
    },
    build: {
      outDir: 'dist',
      sourcemap: mode !== 'production',
      target: 'es2020',
    },
    test: {
      environment: 'jsdom',
      setupFiles: ['./src/test/setup.ts'],
      css: false,
      restoreMocks: true,
    },
  };
});
