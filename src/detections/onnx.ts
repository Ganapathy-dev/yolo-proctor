import * as ort from "onnxruntime-web";

const COCO_CLASSES = ["person","chair","couch","bed","laptop","mouse","keyboard","cell phone","book","backpack","bottle","cup"];

export async function runOnnx(
  video: HTMLVideoElement,
  canvas: HTMLCanvasElement,
  modelSelector: HTMLSelectElement,
  logDiv: HTMLElement
) {
  const ctx = canvas.getContext("2d")!;
  canvas.width = 640;
  canvas.height = 640;

  let session: ort.InferenceSession | null = null;
  let loadingModel = false;
  const FRAME_INTERVAL = 80; // ~12 FPS
  let lastInferenceTime = 0;

  function sigmoid(x: number) { return 1 / (1 + Math.exp(-x)); }

  async function loadModel(path: string) {
    loadingModel = true;
    console.time("Model Load");
    logDiv.textContent = `📥 Loading model: ${path} ...`;
    try {
      session = await ort.InferenceSession.create(path, { executionProviders: ["wasm"] });
      logDiv.textContent = `✅ Loaded model: ${path}`;
    } catch (err) {
      console.error("❌ Failed to load model", err);
      logDiv.textContent = `❌ Failed to load model`;
    } finally {
      loadingModel = false;
      console.timeEnd("Model Load");
    }
  }

  modelSelector.addEventListener("change", async () => {
    await loadModel(modelSelector.value);
  });

  await loadModel(modelSelector.value);

  // Continuous video rendering
  function renderLoop() {
    const t0 = performance.now();
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    const t1 = performance.now();
    console.log(`Render Loop: ${(t1 - t0).toFixed(1)}ms`);
    requestAnimationFrame(renderLoop);
  }
  renderLoop();

  // NMS function
  function nonMaxSuppression(boxes: number[][], scores: number[], iouThreshold = 0.5) {
    const picked: number[] = [];
    const sorted = scores
      .map((score, i) => ({ score, i }))
      .sort((a, b) => b.score - a.score);

    while (sorted.length > 0) {
      const { i } = sorted.shift()!;
      picked.push(i);

      for (let j = sorted.length - 1; j >= 0; j--) {
        const iou = computeIoU(boxes[i], boxes[sorted[j].i]);
        if (iou > iouThreshold) sorted.splice(j, 1);
      }
    }
    return picked;
  }

  function computeIoU(box1: number[], box2: number[]) {
    const [x1, y1, x2, y2] = box1;
    const [x1b, y1b, x2b, y2b] = box2;
    const xi1 = Math.max(x1, x1b);
    const yi1 = Math.max(y1, y1b);
    const xi2 = Math.min(x2, x2b);
    const yi2 = Math.min(y2, y2b);
    const inter = Math.max(0, xi2 - xi1) * Math.max(0, yi2 - yi1);
    const union = (x2 - x1) * (y2 - y1) + (x2b - x1b) * (y2b - y1b) - inter;
    return inter / union;
  }

  async function inferenceLoop() {
    if (!session || loadingModel) {
      setTimeout(inferenceLoop, FRAME_INTERVAL);
      return;
    }

    const now = performance.now();
    if (now - lastInferenceTime < FRAME_INTERVAL) {
      setTimeout(inferenceLoop, FRAME_INTERVAL - (now - lastInferenceTime));
      return;
    }
    lastInferenceTime = now;

    const t0 = performance.now();

    try {
      // Capture pixels
      const t1 = performance.now();
      const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
      console.log(`Pixel Read: ${(t1 - t0).toFixed(1)}ms`);

      // Preprocessing
      const inputData = new Float32Array(3 * canvas.width * canvas.height);
      for (let i = 0; i < canvas.width * canvas.height; i++) {
        inputData[i] = imageData[i*4]/255;
        inputData[i + canvas.width*canvas.height] = imageData[i*4 + 1]/255;
        inputData[i + 2*canvas.width*canvas.height] = imageData[i*4 + 2]/255;
      }
      const t2 = performance.now();
      console.log(`Preprocessing: ${(t2 - t1).toFixed(1)}ms`);

      // Run inference
      const inputTensor = new ort.Tensor("float32", inputData, [1,3,canvas.width,canvas.height]);
      const t3 = performance.now();
      const output = await session.run({ images: inputTensor });
      const t4 = performance.now();
      console.log(`Inference: ${(t4 - t3).toFixed(1)}ms`);

      const key = Object.keys(output)[0];
      const data = (output[key] as ort.Tensor).data as Float32Array;

      // Postprocessing
      const t5 = performance.now();
      const numClasses = 80;
      const numBoxes = 8400;

      const boxes: number[][] = [];
      const scores: number[] = [];
      const classIndices: number[] = [];

      for (let i = 0; i < numBoxes; i++) {
        const offset = i*(numClasses+5);
        const objScore = sigmoid(data[offset+4]);
        if(objScore < 0.5) continue; // ignore low objectness

        let maxScore = 0, maxClass = 0;
        for (let c = 0; c < numClasses; c++) {
          const s = sigmoid(data[offset+5+c]);
          if(s > maxScore){ maxScore = s; maxClass = c; }
        }
        const finalScore = objScore * maxScore;
        if(finalScore < 0.5) continue; // confidence threshold

        // Box coordinates (YOLO usually outputs center-x, center-y, width, height)
        const cx = data[offset];
        const cy = data[offset+1];
        const w = data[offset+2];
        const h = data[offset+3];
        const x1 = cx - w/2;
        const y1 = cy - h/2;
        const x2 = cx + w/2;
        const y2 = cy + h/2;

        boxes.push([x1, y1, x2, y2]);
        scores.push(finalScore);
        classIndices.push(maxClass);
      }

      // Apply NMS
      const picked = nonMaxSuppression(boxes, scores, 0.5);

      let logText = "";
      picked.forEach(i => {
        logText += `${COCO_CLASSES[classIndices[i]]} - ${(scores[i]*100).toFixed(1)}%\n`;
      });

      logDiv.textContent = logText || "(No high confidence detections)";
      const t6 = performance.now();
      console.log(`Postprocessing: ${(t6 - t5).toFixed(1)}ms`);
      console.log(`Total Inference Loop: ${(t6 - t0).toFixed(1)}ms`);

    } catch (err) {
      console.error("ONNX inference error:", err);
    } finally {
      setTimeout(inferenceLoop, 0);
    }
  }

  inferenceLoop();
}
