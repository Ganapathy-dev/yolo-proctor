import { defineConfig } from 'vite'

export default defineConfig({
  server: {

  },
  optimizeDeps: {
    exclude: ['onnxruntime-web'],
  },
})