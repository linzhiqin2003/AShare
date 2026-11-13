import { defineConfig } from 'vitest/config'
import vue from '@vitejs/plugin-vue'

export default defineConfig({
  plugins: [vue()],
  build: {
    outDir: '../src/ashare/platform/static', emptyOutDir: true,
    rollupOptions: { output: { manualChunks: id => /node_modules\/(?:@codemirror|@lezer)\//.test(id) ? 'editor' : undefined } },
  },
  server: { host: '127.0.0.1', proxy: { '/api': 'http://127.0.0.1:8765' } },
  test: { setupFiles: ['./src/test-setup.ts'], environment: 'jsdom', testTimeout: 10000 },
})
