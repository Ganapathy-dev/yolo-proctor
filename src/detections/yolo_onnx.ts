const COCO_CLASSES = ["person","chair","couch","bed","laptop","mouse","keyboard","cell phone","book","backpack","bottle","cup"];

import * as ort from "onnxruntime-web";
import { initCamera } from "../utils";

const video = document.getElementById("webcam") as HTMLVideoElement;
const uiLog = document.getElementById("log")!;
const modelSelector = document.getElementById("modelSelector") as HTMLSelectElement;

const canvas = document.createElement("canvas");
const ctx = canvas.getContext("2d")!;
canvas.width = 640;
canvas.height = 640;

let session: ort.InferenceSession | null = null;
let loadingModel = false;

function sigmoid(x: number) {
  return 1 / (1 + Math.exp(-x));
}

async function loadModel(path: string) {
  loadingModel = true;

  console.log(`📥 Loading model: ${path}`);
  uiLog.textContent = `Loading model: ${path} ...`;

  session = await ort.InferenceSession.create(path, {
    executionProviders: ["wasm"]
  });

  console.log(`✅ Model loaded successfully: ${path}`);
  uiLog.textContent = `Loaded model: ${path}`;

  loadingModel = false;
}

modelSelector.addEventListener("change", async () => {
  await loadModel(modelSelector.value);
});

async function main() {
  console.log("🎥 Initializing camera...");
  await initCamera(video);
  console.log("✔ Camera ready:", video.videoWidth, "x", video.videoHeight);

  console.log("🔌 Loading initial model...");
  await loadModel(modelSelector.value);

  const numClasses = 80;
  const numBoxes = 8400;

  let lastTime = performance.now();

  async function detectFrame() {
    requestAnimationFrame(detectFrame);

    if (!session || loadingModel) return;

    const now = performance.now();
    if (now - lastTime < 80) return;
    lastTime = now;

    try {
      console.log("📸 Capturing frame...");
      const t0 = performance.now();

      ctx.drawImage(video, 0, 0, 640, 640);
      const t1 = performance.now();
      console.log("✔ Frame captured");

      console.log("🧪 Preprocessing...");
      const imageData = ctx.getImageData(0, 0, 640, 640).data;

      const inputData = new Float32Array(3 * 640 * 640);
      for (let i = 0; i < 640 * 640; i++) {
        inputData[i] = imageData[i * 4] / 255;
        inputData[i + 409600] = imageData[i * 4 + 1] / 255;
        inputData[i + 819200] = imageData[i * 4 + 2] / 255;
      }
      const t2 = performance.now();
      console.log("✔ Preprocessing completed");

      console.log("⚡ Running inference...");
      const inputTensor = new ort.Tensor("float32", inputData, [1, 3, 640, 640]);
      const output = await session.run({ images: inputTensor });
      const t3 = performance.now();
      console.log("✔ Inference finished");

      console.log("📝 Postprocessing results...");
      const key = Object.keys(output)[0];
      const data = (output[key] as ort.Tensor).data as Float32Array;

      let logText = "";

      for (let i = 0; i < numBoxes; i++) {
        const offset = i * (numClasses + 5);
        const objScore = sigmoid(data[offset + 4]);

        let maxScore = 0;
        let maxClass = 0;

        for (let c = 0; c < numClasses; c++) {
          const s = sigmoid(data[offset + 5 + c]);
          if (s > maxScore) {
            maxScore = s;
            maxClass = c;
          }
        }

        const finalScore = objScore * maxScore;

        if (finalScore >= 0.90) {
          logText += `${COCO_CLASSES[maxClass]} - ${(finalScore * 100).toFixed(1)}%\n`;
        }
      }

      const t4 = performance.now();
      console.log("✔ Postprocessing completed");

      uiLog.textContent = logText || "(No high confidence detections)";

      console.log(
        `⏱ Frame ${(t4 - t0).toFixed(1)}ms | ` +
        `capture ${(t1 - t0).toFixed(1)}ms | preprocess ${(t2 - t1).toFixed(1)}ms | ` +
        `infer ${(t3 - t2).toFixed(1)}ms | post ${(t4 - t3).toFixed(1)}ms`
      );

    } catch (err) {
      console.error("❌ ERROR in detectFrame()", err);
    }
  }

  detectFrame();
}

main();
