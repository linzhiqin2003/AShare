import { defineConfig } from 'vitest/config'
import vue from '@vitejs/plugin-vue'

export default defineConfig({
  plugins: [vue()],
  build: {
    outDir: '../src/ashare/platform/static', emptyOutDir: true,
    rollupOptions: { output: { manualChunks: id => /node_modules\/(?:@codemirror|@lezer)\//.test(id) ? 'editor' : undefined } },
  },
  // 代理必须保留浏览器的 Host：后端要求 Origin 与 Host 一致，Vite 的字符串简写会改写 Host。
  server: { host: '127.0.0.1', proxy: { '/api': { target: 'http://127.0.0.1:8765', changeOrigin: false } } },
  test: { setupFiles: ['./src/test-setup.ts'], environment: 'jsdom', testTimeout: 10000 },
})
