import { defineConfig } from 'vite'
import path from "path";

export default defineConfig({
  base: "/yolo-proctor/",
  server: {},
  build: {
    rollupOptions: {
      input: {
        main: path.resolve(__dirname, "index.html"),
        temp: path.resolve(__dirname, "temp.html"),
      },
    },
  },
  optimizeDeps: {
    exclude: ['onnxruntime-web'],
  },
})