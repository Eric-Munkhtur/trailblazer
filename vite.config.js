import { defineConfig, loadEnv } from 'vite';

export default defineConfig(({ mode }) => {
  const env = { ...loadEnv(mode, process.cwd(), ''), ...process.env };
  return {
    build: { chunkSizeWarningLimit: 1800 },
    server: {
      port: 5173,
      proxy: { '/api': `http://localhost:${Number(env.PORT) || 8790}` },
    },
  };
});
