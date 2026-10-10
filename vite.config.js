import { defineConfig } from 'vite';

export default defineConfig({
  build: { chunkSizeWarningLimit: 1800 },
  server: {
    port: 5173,
    proxy: { '/api': 'http://localhost:8787' },
  },
});
