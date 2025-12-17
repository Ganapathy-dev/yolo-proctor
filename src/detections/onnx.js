import cv from "@techstark/opencv-js";
import * as ort from "onnxruntime-web";
import { Tensor } from "onnxruntime-web";
import { renderBoxes } from "./renderBox.js";



let yolov8Session = null;
let nmsSession = null;

export async function initYoloOnnx() {
  if (!yolov8Session) {
    yolov8Session = await ort.InferenceSession.create("./yolov8n.onnx", {
      executionProviders: ["wasm"]
    });
  }

  if (!nmsSession) {
    nmsSession = await ort.InferenceSession.create("./nms-yolov8.onnx", {
      executionProviders: ["wasm"]
    });
  }

  return { yolov8Session, nmsSession };
}


export async function runOnnxWebcam(sessions, video, canvas) {

  const ctx = canvas.getContext("2d");
  canvas.width = 640;
  canvas.height = 640;
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

  await detectImage(canvas, canvas, sessions, 100, 0.45, 0.25, [1, 3, 640, 640]);

  requestAnimationFrame(() => runOnnxWebcam(sessions, video, canvas));
}


export async function detectImage(image, canvas, sessions, topk, iouThreshold, scoreThreshold, inputShape) {
  console.log("Entering detect image");
  const { yolov8Session, nmsSession } = sessions;
  const [modelWidth, modelHeight] = inputShape.slice(2);
  const [input, xRatio, yRatio] = preprocessing(image, modelWidth, modelHeight);

  const tensor = new Tensor("float32", input.data32F, inputShape);
  const config = new Tensor("float32", new Float32Array([topk, iouThreshold, scoreThreshold]));

  const { output0 } = await yolov8Session.run({ images: tensor });

  const { selected } = await nmsSession.run({ detection: output0, config });

  const boxes = [];

  for (let idx = 0; idx < selected.dims[1]; idx++) {
    const data = selected.data.slice(idx * selected.dims[2], (idx + 1) * selected.dims[2]);
    const box = data.slice(0, 4);
    const scores = data.slice(4);
    const score = Math.max(...scores);
    const label = scores.indexOf(score);

    const [x, y, w, h] = [
      (box[0] - 0.5 * box[2]) * xRatio,
      (box[1] - 0.5 * box[3]) * yRatio,
      box[2] * xRatio,
      box[3] * yRatio
    ];

    boxes.push({ label, probability: score, bounding: [x, y, w, h] });
  }

  console.log("boxes", boxes);

  renderBoxes(canvas, boxes);
  input.delete();
}

function preprocessing(source, modelWidth, modelHeight) {
  console.log("Entering preporcessing ");
  const mat = cv.imread(source);
  const matC3 = new cv.Mat(mat.rows, mat.cols, cv.CV_8UC3);
  cv.cvtColor(mat, matC3, cv.COLOR_RGBA2BGR);

  const maxSize = Math.max(matC3.rows, matC3.cols);
  const xPad = maxSize - matC3.cols,
    xRatio = maxSize / matC3.cols;
  const yPad = maxSize - matC3.rows,
    yRatio = maxSize / matC3.rows;
  const matPad = new cv.Mat();
  cv.copyMakeBorder(matC3, matPad, 0, yPad, 0, xPad, cv.BORDER_CONSTANT);

  const input = cv.blobFromImage(
    matPad,
    1 / 255.0,
    new cv.Size(modelWidth, modelHeight),
    new cv.Scalar(0, 0, 0),
    true,
    false
  );

  mat.delete();
  matC3.delete();
  matPad.delete();

  return [input, xRatio, yRatio];
}
