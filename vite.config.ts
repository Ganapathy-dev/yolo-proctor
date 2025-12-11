import { defineConfig } from 'vite'

export default defineConfig({
  base: "/yolo-proctor/",
  server: {

  },
  optimizeDeps: {
    exclude: ['onnxruntime-web'],
  },
})