import cv from "@techstark/opencv-js";
import { Tensor, InferenceSession } from "onnxruntime-web";
import labels from "./labels.json";

export interface YoloSession {
  net: InferenceSession;
  nms: InferenceSession;
}

export interface DetectionBox {
  label: number;
  probability: number;
  bounding: [number, number, number, number]; // [x, y, w, h] in original image coords
}

const MODEL_INPUT_SHAPE = [1, 3, 640, 640];
let session: YoloSession | null = null;
let cvReady = false;
let initializing = false;

let lastBoxes: DetectionBox[] = [];
let webcamRunning = false;
let lastDetectTime = 0;
const loggedLabelMap: Map<number, HTMLParagraphElement> = new Map();

// ---------------- Debug & Logging ----------------
// const dbg = (...args: any[]) => console.log("[ONNX-DEBUG]", ...args);
const logDetection = (msg: string) => {
  const logDiv = document.getElementById("log");
  if (logDiv) {
    const p = document.createElement("p");
    p.textContent = msg;
    logDiv.appendChild(p);
  }
};

// ---------------- OpenCV Load ----------------
const loadOpenCV = (): Promise<void> => {
  return new Promise(resolve => {
    if (cvReady) return resolve();
    if ((cv as any).getBuildInformation) {
      cvReady = true;
      return resolve();
    }
    cv.onRuntimeInitialized = () => {
      cvReady = true;
      resolve();
    };
  });
};

// ---------------- Preprocessing (Letterbox) ----------------
const letterbox = (
  mat: cv.Mat,
  targetWidth: number,
  targetHeight: number
): { blob: cv.Mat; xScale: number; yScale: number; dx: number; dy: number } => {
  const origW = mat.cols;
  const origH = mat.rows;

  const scale = Math.min(targetWidth / origW, targetHeight / origH);
  const newW = Math.round(origW * scale);
  const newH = Math.round(origH * scale);

  const dx = Math.floor((targetWidth - newW) / 2);
  const dy = Math.floor((targetHeight - newH) / 2);

  const resized = new cv.Mat();
  cv.resize(mat, resized, new cv.Size(newW, newH));

  const padded = new cv.Mat();
  cv.copyMakeBorder(
    resized,
    padded,
    dy,
    targetHeight - newH - dy,
    dx,
    targetWidth - newW - dx,
    cv.BORDER_CONSTANT,
    new cv.Scalar(114, 114, 114)
  );

  const blob = cv.blobFromImage(padded, 1 / 255.0, new cv.Size(targetWidth, targetHeight), new cv.Scalar(0, 0, 0), true, false);

  resized.delete();
  padded.delete();

  return { blob, xScale: scale, yScale: scale, dx, dy };
};

// ---------------- Detection ----------------
export const detectImage = async (
  input: HTMLImageElement | HTMLCanvasElement,
  sess: YoloSession
): Promise<DetectionBox[]> => {
  const mat = cv.imread(input as any);
  const matC3 = new cv.Mat();
  cv.cvtColor(mat, matC3, cv.COLOR_RGBA2BGR);

  const { blob, xScale, yScale, dx, dy } = letterbox(matC3, MODEL_INPUT_SHAPE[2], MODEL_INPUT_SHAPE[3]);
  mat.delete();
  matC3.delete();

  const tensor = new Tensor("float32", blob.data32F, MODEL_INPUT_SHAPE);
  blob.delete();

  const config = new Tensor("float32", new Float32Array([100, 0.45, 0.25]));
  const { output0 } = await sess.net.run({ images: tensor });
  const { selected } = await sess.nms.run({ detection: output0, config });

  const boxes: DetectionBox[] = [];
  const data = selected.data as Float32Array;

  for (let i = 0; i < selected.dims[1]; i++) {
    const offset = i * selected.dims[2];
    const row = data.subarray(offset, offset + selected.dims[2]);
    const box = row.subarray(0, 4);
    const scores = row.subarray(4);

    let maxScore = -Infinity;
    let label = -1;
    for (let j = 0; j < scores.length; j++) {
      if (scores[j] > maxScore) {
        maxScore = scores[j];
        label = j;
      }
    }

    // Map box back to original image coords
    const x = (box[0] - box[2] / 2 - dx) / xScale;
    const y = (box[1] - box[3] / 2 - dy) / yScale;
    const w = box[2] / xScale;
    const h = box[3] / yScale;

    boxes.push({ label, probability: maxScore, bounding: [x, y, w, h] });
  }

  return boxes;
};

// ---------------- ONNX Model Init ----------------
const initOnnx = async (modelPath: string) => {
  if (session) return session;
  if (initializing) {
    while (!session) await new Promise(r => setTimeout(r, 50));
    return session;
  }

  initializing = true;
  await loadOpenCV();

  const net = await InferenceSession.create(modelPath, { executionProviders: ["webgpu"] });
  const nms = await InferenceSession.create("/yolo-proctor/nms-yolov8.onnx");

  // Warmup
  const dummy = new Tensor("float32", new Float32Array(MODEL_INPUT_SHAPE.reduce((a, b) => a * b)), MODEL_INPUT_SHAPE);
  await net.run({ images: dummy });

  session = { net, nms };
  initializing = false;
  return session;
};

// ---------------- Draw Boxes ----------------
export const drawYoloPredictions = (boxes: DetectionBox[], ctx: CanvasRenderingContext2D) => {
  boxes.forEach(box => {
    const [x, y, w, h] = box.bounding;
    const labelName = labels[box.label] || "unknown";

    ctx.strokeStyle = "lime";
    ctx.lineWidth = 2;
    ctx.strokeRect(x, y, w, h);

    ctx.font = "16px Arial";
    ctx.fillStyle = "lime";
    ctx.fillText(`${labelName} (${(box.probability * 100).toFixed(1)}%)`, x, y - 5);
  });
};

// ---------------- Image Detection ----------------
export const runOnnxImage = async (image: HTMLImageElement, canvas: HTMLCanvasElement, modelPath: string) => {
  if (!image.complete || image.naturalWidth === 0) return logDetection("❌ Image not loaded");

  const sess = await initOnnx(modelPath);
  canvas.width = image.naturalWidth;
  canvas.height = image.naturalHeight;

  const ctx = canvas.getContext("2d")!;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(image, 0, 0);

  const boxes = await detectImage(image, sess);

  if (!boxes.length) logDetection("No objects detected");
  else {
    boxes.forEach((b, i) => logDetection(`Box ${i}: ${labels[b.label]} ${b.probability.toFixed(2)}`));
    drawYoloPredictions(boxes, ctx);
  }
};

// ---------------- Webcam Detection ----------------
export const runOnnxWebcam = async (
  video: HTMLVideoElement,
  canvas: HTMLCanvasElement,
  modelPath: string,
  interval = 200
) => {
  if (!video.srcObject) return console.error("Webcam not started");

  const sess = await initOnnx(modelPath);
  const ctx = canvas.getContext("2d")!;
  const tempCanvas = document.createElement("canvas");
  const tempCtx = tempCanvas.getContext("2d")!;
  webcamRunning = true;

  const process = async (ts: number) => {
    if (!webcamRunning) return;

    if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      tempCanvas.width = video.videoWidth;
      tempCanvas.height = video.videoHeight;
    }

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    if (lastBoxes.length) drawYoloPredictions(lastBoxes, ctx);

    if (ts - lastDetectTime >= interval) {
      lastDetectTime = ts;
      tempCtx.drawImage(video, 0, 0, tempCanvas.width, tempCanvas.height);
      const boxes = await detectImage(tempCanvas, sess);
      lastBoxes = boxes;

      const logDiv = document.getElementById("log");
      if (!logDiv) return;

      // Track currently detected labels
      const currentLabels = new Set<number>();
      boxes.forEach((b) => currentLabels.add(b.label));

      // Remove logs for labels no longer detected
      for (const [label, p] of loggedLabelMap.entries()) {
        if (!currentLabels.has(label)) {
          p.remove();
          loggedLabelMap.delete(label);
        }
      }

      // Update/add logs for currently detected labels
      boxes.forEach((b) => {
        const labelName = labels[b.label] || "unknown";
        const text = `Label=${labelName} Confidence=${(b.probability * 100).toFixed(1)}%`;

        if (loggedLabelMap.has(b.label)) {
          // Update existing paragraph
          const p = loggedLabelMap.get(b.label)!;
          p.textContent = text;
        } else {
          // Create new paragraph for this label
          const p = document.createElement("p");
          p.textContent = text;
          logDiv.appendChild(p);
          loggedLabelMap.set(b.label, p);
        }
      });
    }

    requestAnimationFrame(process);
  };

  requestAnimationFrame(process);
};


export const stopOnnxWebcam = () => { webcamRunning = false; };
